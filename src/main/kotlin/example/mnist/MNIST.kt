package example.mnist

import core.tensor.JvmTensor
import core.tensor.Tensor
import core.tensor.Tensors
import kotlin.math.absoluteValue
import kotlin.random.Random
import tools.ImageUtil
import tools.RandomUtil

// https://github.com/tinygrad/tinygrad/blob/91a352a8e2697828a4b1eafa2bdc1a9a3b7deffa/test/mnist.py

private var Rand = Random(123)

infix fun Float.closeTo(other: Float): Boolean {
    return (this - other).absoluteValue < 0.01F
}

data class SGD(val tensors: List<Tensor>, val lr: Float) {
    fun step() {
        for (t in tensors) {
            t as JvmTensor
            val grad = t.grad!! as JvmTensor
            t.data = t.data.sub(grad.data.mul(lr))
        }
    }
}

class MnistNet(rand: Random = Rand) {
    val l1 =
        Tensors.create(
            data = RandomUtil.xavierInitArray(784 * 128, rand),
            shape = intArrayOf(784, 128),
            requiresGrad = true)
    val l2 =
        Tensors.create(
            data = RandomUtil.xavierInitArray(128 * 10, rand),
            shape = intArrayOf(128, 10),
            requiresGrad = true)

    fun forward(x: Tensor): Tensor {
        return x.matmul(l1).relu().matmul(l2).logSoftmax()
    }
}

class MNIST(private val rand: Random = Rand) {
    private val model = MnistNet(rand)
    private val optim = SGD(listOf(model.l1, model.l2), lr = 0.01F)

    private val train = MnistDataSupplier(true)
    private val test = MnistDataSupplier(false)

    fun train(BS: Int = 128) {
        for (i in 0 until 2000) {
            val sampIdxes = RandomUtil.randInt(BS, 0 until train.size(), rand)

            val (xList, yList) = sampIdxes.map(train::get).unzip()

            val X = Tensors.stack(*xList.toTypedArray())
            val Y = Tensors.stack(*yList.toTypedArray())

            val outs = model.forward(X)

            // NLL loss
            val y = Y.mul(Tensors.createScalar(-1F))
            val loss = outs.mul(y).mean()

            loss.backward()
            optim.step()

            val accuracy =
                Tensors.perform(outs.argmax(1), Y.argmax(1)) { a, b -> if (a closeTo b) 1F else 0F }
                    .mean()

            println("iteration $i: loss ${loss.asScalar()}, accuracy ${accuracy.asScalar()}")
        }
    }

    fun eval() {
        val (xList, yList) = (0 until test.size()).map(train::get).unzip()

        val X = Tensors.stack(*xList.toTypedArray())
        val Y = Tensors.stack(*yList.toTypedArray())

        val outs = model.forward(X)

        val pred = outs.argmax(1)
        val real = Y.argmax(1)
        val accuracies = Tensors.perform(pred, real) { a, b -> if (a closeTo b) 1F else 0F }

        println("accuracy on test data set ${accuracies.mean().asScalar()}")

        var count = 10
        for (i in 0 until accuracies.size()) {
            if (accuracies[i] == 0F) {
                ImageUtil.printGrayScaleImage(xList[i].reshape(28, 28).toMatrix())
                println("expect: ${real[i]}, predict: ${pred[i]}")

                count--
                if (count == 0) {
                    break
                }
            }
        }
    }
}

fun main() {
    val mnist = MNIST()
    mnist.train()
    mnist.eval()
}
