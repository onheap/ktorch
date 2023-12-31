package example.mnist

import core.tensor.JvmTensor
import core.tensor.Tensor
import java.io.IOException
import java.io.RandomAccessFile
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.util.zip.GZIPInputStream

private val MNIST_DIR = Path.of("mnist")

// https://github.com/mikex86/scicore/blob/master/tests/src/test/java/me/mikex86/scicore/tests/mnist/MnistDataSupplier.kt
class MnistDataSupplier(train: Boolean, shuffle: Boolean) {
    private val imagesRAF: RandomAccessFile
    private val labelsRAF: RandomAccessFile

    val X = mutableListOf<Tensor>()
    val Y = mutableListOf<Tensor>()

    companion object {
        init {
            downloadMnist()
        }
    }

    init {
        val imagesPath =
            MNIST_DIR.resolve(if (train) "train-images-idx3-ubyte" else "t10k-images-idx3-ubyte")
        val labelsPath =
            MNIST_DIR.resolve(if (train) "train-labels-idx1-ubyte" else "t10k-labels-idx1-ubyte")
        try {
            imagesRAF = RandomAccessFile(imagesPath.toFile(), "r")
            labelsRAF = RandomAccessFile(labelsPath.toFile(), "r")
            val imagesMagic = imagesRAF.readInt()
            val nImages = imagesRAF.readInt()
            val labelsMagic = labelsRAF.readInt()
            val nLabels = labelsRAF.readInt()

            val height = imagesRAF.readInt()
            val width = imagesRAF.readInt()

            if (imagesMagic != 2051 || labelsMagic != 2049) {
                throw IOException("Invalid MNIST file")
            }
            if (nImages != nLabels) {
                throw IOException("Images and labels have different number of samples")
            }

            val imageData =
                ByteBuffer.allocateDirect(height * width * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until nImages) {
                val label = labelsRAF.readByte()
                val bytes = ByteArray(height * width)
                imagesRAF.read(bytes)

                for (j in 0 until height * width) {
                    imageData.putFloat((bytes[j].toInt() and 0xFF) / 255.0f)
                }

                val bf = imageData.flip().asFloatBuffer()
                val data = FloatArray(height * width)
                bf.get(data)

                X.add(Tensor.create(data))
                Y.add(Tensor.zeros(10).also { it[label.toInt()] = 1F })
            }
        } catch (e: IOException) {
            throw RuntimeException(e)
        }
    }

    fun get(idx: Int): Pair<Tensor, Tensor> {
        return X[idx] to Y[idx]
    }
}

fun main() {
    val s = MnistDataSupplier(true, false)
    val (x, y) = s.get(9)
    println((x as JvmTensor).data.reshape(28, 28).performElementwise { if (it == 0F) 1F else 8F })
    println((y as JvmTensor).data)
}

@Throws(IOException::class, InterruptedException::class)
private fun downloadMnist() {
    if (MNIST_DIR.toFile().exists()) {
        println("MNIST already downloaded")
        return
    }
    val client = HttpClient.newHttpClient()
    val urls =
        listOf(
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")
    Files.createDirectories(MNIST_DIR)
    for (url in urls) {
        val filename = url.substring(url.lastIndexOf('/') + 1)
        val request = HttpRequest.newBuilder().uri(URI.create(url)).GET().build()
        val path = MNIST_DIR.resolve(filename)
        client.send(request, HttpResponse.BodyHandlers.ofFile(path))

        // inflate gz files
        if (filename.endsWith(".gz")) {
            val `in` = GZIPInputStream(Files.newInputStream(path))
            Files.copy(`in`, path.resolveSibling(filename.substring(0, filename.length - 3)))
        }
    }
}
