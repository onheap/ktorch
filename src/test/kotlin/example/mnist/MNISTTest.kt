package example.mnist

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import core.tensor.JvmTensor
import core.tensor.Tensor
import core.tensor.Tensors
import kotlin.random.Random
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import tools.*

fun NDArray.logSoftmax(): NDArray = this.logSoftmax(-1)

fun NDArray.relu(): NDArray = Activation.relu(this)

class MNISTTest {

    private lateinit var manager: NDManager
    private lateinit var rand: Random

    private val train = MnistDataSupplier(true)
    private val test = MnistDataSupplier(false)

    private lateinit var l1n: NDArray
    private lateinit var l2n: NDArray

    private lateinit var l1t: Tensor
    private lateinit var l2t: Tensor

    private val lr = 0.01F
    private val BS = 128

    private fun forward(x: NDArray): NDArray {
        return x.matMul(l1n).relu().matMul(l2n).logSoftmax()
    }

    private fun forward(x: Tensor): Tensor {
        return x.matmul(l1t).relu().matmul(l2t).logSoftmax()
    }

    private fun sgd(vararg ndarrays: NDArray) {
        for (n in ndarrays) {
            n.gradient.mul(lr).use { diff -> n.subi(diff) }
        }
    }

    private fun sgd(vararg tensors: Tensor) {
        for (t in tensors) {
            t as JvmTensor
            val grad = t.grad!! as JvmTensor
            t.data = t.data.sub(grad.data.mul(lr))
        }
    }

    @BeforeEach
    fun setUp() {
        manager = NDManager.newBaseManager()
        rand = Random(123)

        val l1Data = RandomUtil.xavierInitArray(784 * 128, rand)
        val l2Data = RandomUtil.xavierInitArray(128 * 10, rand)

        l1n = manager.create(l1Data, Shape(784, 128)).also { it.setRequiresGradient(true) }
        l2n = manager.create(l2Data, Shape(128, 10)).also { it.setRequiresGradient(true) }

        l1t = Tensors.create(data = l1Data, shape = intArrayOf(784, 128), requiresGrad = true)
        l2t = Tensors.create(data = l2Data, shape = intArrayOf(128, 10), requiresGrad = true)
    }

    @AfterEach
    fun tearDown() {
        manager.close()
    }

    @Test
    fun test() {
        ProgressBar.loopFor(60000) {
            //        for (i in 0 until 2000) {
            val sampIdxes = RandomUtil.randInt(BS, 0 until train.size(), rand)

            manager.newSubManager().use { subManager ->
                val (xArray, yArray) =
                    sampIdxes.map(train::getArray).unzip().let { (x, y) ->
                        x.toTypedArray() to y.toTypedArray()
                    }

                val X = subManager.create(xArray)
                val Y = subManager.create(yArray)

                val (outs, loss) =
                    subManager.engine.newGradientCollector().use { gc ->
                        val outs = forward(X)

                        // NLL loss
                        val y = Y.mul(-1F)
                        val loss = outs.mul(y).mean()

                        gc.backward(loss)
                        return@use outs to loss
                    }

                sgd(l1n, l2n)

                val accuracy =
                    outs.argMax(-1).eq(Y.argMax(-1)).toType(DataType.FLOAT32, true).mean()

                ("loss ${loss.getFloat()}, accuracy ${accuracy.getFloat()}")
            }
        }
    }

    @Test
    fun testMnistCorrectness() {
        // verify mnist step by step
        for (i in 0 until 100) {
            val sampIdxes = RandomUtil.randInt(BS, 0 until train.size(), rand)

            val (xArray, yArray) =
                sampIdxes.map(train::getArray).unzip().let { (x, y) ->
                    x.toTypedArray() to y.toTypedArray()
                }

            val subManager = manager.newSubManager()

            val XN = subManager.create(xArray)
            val YN = subManager.create(yArray)

            val XT = Tensors.create(xArray)
            val YT = Tensors.create(yArray)

            val (outsn, lossn) =
                (XN to YN).let { (X, Y) ->
                    subManager.engine.newGradientCollector().use { gc ->
                        val outs = forward(X)

                        // NLL loss
                        val y = Y.mul(-1F)
                        val loss = outs.mul(y).mean()

                        gc.backward(loss)
                        return@use outs to loss
                    }
                }

            val (outst, losst) =
                (XT to YT).let { (X, Y) ->
                    val outs = forward(X)

                    // NLL loss
                    val y = Y.mul(Tensors.createScalar(-1F))
                    val loss = outs.mul(y).mean()

                    loss.backward()
                    return@let outs to loss
                }

            assertTensorEquals(outsn, outst)
            assertTensorEquals(lossn, losst)

            assertTensorEquals(l1n, l1t)
            assertTensorEquals(l2n, l2t)
            assertTensorEquals(l1n.gradient, l1t.grad!!)
            assertTensorEquals(l2n.gradient, l2t.grad!!)

            val accuracy =
                Tensors.perform(outst.argmax(1), YT.argmax(1)) { a, b ->
                        if (a closeTo b) 1F else 0F
                    }
                    .mean()

            println(
                "iteration $i: accuracy ${accuracy.asScalar()}\n loss ndarray: ${lossn.getFloat()}, tensor: ${losst.asScalar()}")

            sgd(l1n, l2n)
            sgd(l1t, l2t)

            subManager.close()
        }

        val (xArray, yArray) =
            (0 until test.size()).map(test::getArray).unzip().let { (x, y) ->
                x.toTypedArray() to y.toTypedArray()
            }

        val XN = manager.create(xArray)
        val YN = manager.create(yArray)

        val XT = Tensors.create(xArray)
        val YT = Tensors.create(yArray)

        val outsn = forward(XN)
        val outst = forward(XT)

        assertTensorEquals(outsn, outst)

        val accuracy =
            Tensors.perform(outst.argmax(1), YT.argmax(1)) { a, b -> if (a closeTo b) 1F else 0F }
                .mean()

        println("accuracy on test data set ${accuracy.asScalar()}")
    }
}

fun main(args: Array<String>) {
    val test = MNISTTest()
    test.setUp()
    test.test()
    test.tearDown()
}
