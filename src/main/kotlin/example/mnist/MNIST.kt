package example.mnist

import core.tensor.JvmTensor
import core.tensor.Tensor
import kotlin.math.absoluteValue
import kotlin.math.sqrt
import kotlin.random.Random
import ndarray.NDArray

// https://github.com/tinygrad/tinygrad/blob/91a352a8e2697828a4b1eafa2bdc1a9a3b7deffa/test/mnist.py

infix fun Float.closeTo(other: Float): Boolean {
    return (this - other).absoluteValue < 0.01F
}

fun layerInit(inputSize: Int, outputSize: Int): Tensor {
    return Tensor.create(
        data =
            FloatArray(inputSize * outputSize) {
                (Random.nextFloat() * 2 - 1) / sqrt(inputSize.toFloat() * outputSize)
            },
        shape = intArrayOf(inputSize, outputSize),
        requiresGrad = true)
}

data class SGD(val tensors: List<JvmTensor>, val lr: JvmTensor) {
    fun step() {
        for (t in tensors) {
            //   t.data -= self.lr * t.grad
            val grad = t.grad!! as JvmTensor
            t.data = t.data.sub(grad.data.mul(lr.data))
        }
    }
}

class MnistNet {
    val l1 = layerInit(784, 128) as JvmTensor

    val l2 = layerInit(128, 10) as JvmTensor

    fun forward(x: Tensor): JvmTensor {
        return x.matmul(l1).relu().matmul(l2).logSoftmax() as JvmTensor
    }
}

val model = MnistNet()
val optim = SGD(listOf(model.l1, model.l2), lr = Tensor.createScalar(0.01F) as JvmTensor)

val train = MnistDataSupplier(true)
val test = MnistDataSupplier(false)

fun train(BS: Int = 128) {
    for (i in 0 until 2000) {
        val sampIdxes = IntArray(BS) { Random.nextInt(0, train.size()) }

        val (xList, yList) =
            sampIdxes
                .map(train::get)
                .map { (x, y) -> (x as JvmTensor).data to (y as JvmTensor).data }
                .unzip()

        val X = JvmTensor(data = NDArray.merge(*xList.toTypedArray()), requiresGrad = true)
        val Y = JvmTensor(data = NDArray.merge(*yList.toTypedArray()), requiresGrad = true)

        val outs = model.forward(X)

        // NLL loss
        val y = Y.mul(Tensor.createScalar(-1F))
        val loss = outs.mul(y).mean()

        loss.backward()
        optim.step()

        val accuracy =
            NDArray.perform(outs.data.argmax(1), Y.data.argmax(1)) { a, b ->
                    if (a closeTo b) 1F else 0F
                }
                .mean()

        println("iteration $i: loss ${loss.asScalar()}, accuracy ${accuracy.asScalar()}")
    }
}

fun eval() {
    val xList = mutableListOf<NDArray>()
    val yList = mutableListOf<NDArray>()

    for (i in 0 until test.size()) {
        val (x, y) = train.get(i).let { (x, y) -> (x as JvmTensor).data to (y as JvmTensor).data }
        xList.add(x)
        yList.add(y)
    }

    val X = JvmTensor(data = NDArray.merge(*xList.toTypedArray()), requiresGrad = true)
    val Y = JvmTensor(data = NDArray.merge(*yList.toTypedArray()), requiresGrad = true)

    val outs = model.forward(X)

    val accuracy =
        NDArray.perform(outs.data.argmax(1), Y.data.argmax(1)) { a, b ->
                if (a closeTo b) 1F else 0F
            }
            .mean()

    println("accuracy on test data set ${accuracy.asScalar()}")
}

fun main() {
    train()
    eval()
}
