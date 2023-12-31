package core.tensor

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.nn.Activation
import ai.djl.training.GradientCollector
import kotlin.random.Random
import ndarray.Util.arrOf
import ndarray.Util.arrOfF
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import tools.*

fun NDArray.logSoftmax(): NDArray = this.logSoftmax(-1)

fun NDArray.relu(): NDArray = Activation.relu(this)

class JvmTensorTest {

    private lateinit var manager: NDManager
    private lateinit var gc: GradientCollector

    @BeforeEach
    fun setUp() {
        manager = NDManager.newBaseManager()
        gc = manager.engine.newGradientCollector()
    }

    @AfterEach
    fun tearDown() {
        manager.close()
        gc.close()
    }

    @Test
    fun test() {
        val fa = arrOfF(1, 1)
        val fb = arrOfF(2, 2)

        val sa = arrOf(2)
        val sb = arrOf(1, 2)

        val a1 = manager.create(data = fa, shape = sa, requiresGrad = true)
        val b1 = manager.create(data = fb, shape = sb, requiresGrad = true)

        (a1 to b1).let { (a, b) ->
            val c = a.add(b) // 3
            val res = c

            gc.backward(res)

            printMessage("==============")
            printObjects(true, "res1", res)
            printObjects(true, "a1 grad", a.gradient)
            printObjects(true, "b1 grad", b.gradient)
        }

        val a2 = Tensor.create(data = fa, shape = sa, requiresGrad = true)
        val b2 = Tensor.create(data = fb, shape = sb, requiresGrad = true)

        (a2 to b2).let { (a, b) ->
            val c = a.add(b) // 3
            val res = c

            res.backward()

            printMessage("==============")
            printObjects(true, "res2", res)
            printObjects(true, "a2 grad", a2.grad)
            printObjects(true, "b2 grad", b2.grad)
        }
    }

    @Test
    fun testArithmeticOperatorsCorrectness() {
        repeat(100) {
            val m = Random.nextInt(1, 128)
            val n = Random.nextInt(1, 128)

            if (it % 10 == 0) {
                print(it)
                printMessage(" A: $m X $n, B: $m X $n")
            }

            val fa = FloatArray(m * n) { randomFloat() }
            val fb = FloatArray(m * n) { randomFloat(excludeZero = true) }

            val shape = intArrayOf(m, n)

            val ta = Tensor.create(data = fa, shape = shape, requiresGrad = true)
            val tb = Tensor.create(data = fb, shape = shape, requiresGrad = true)

            val da = manager.create(data = fa, shape = shape, requiresGrad = true)
            val db = manager.create(data = fb, shape = shape, requiresGrad = true)

            assertOpResEqual(BOp(da, db, NDArray::add), BOp(ta, tb, Tensor::add))
            assertOpResEqual(BOp(da, db, NDArray::mul), BOp(ta, tb, Tensor::mul))
            assertOpResEqual(UOp(da, NDArray::logSoftmax), UOp(ta, Tensor::logSoftmax))
            assertOpResEqual(UOp(da, NDArray::relu), UOp(ta, Tensor::relu))
            assertOpResEqual(UOp(da, NDArray::sum), UOp(ta, Tensor::sum), 0.1F)

            assertOpResEqual(
                BOp(da, db) { a, b ->
                    val c = a.add(b)
                    val d = c.mul(c)
                    val e = d.add(a)
                    val h = e.add(b)
                    val i = h.relu()
                    return@BOp i.logSoftmax()
                },
                BOp(ta, tb) { a, b ->
                    val c = a.add(b)
                    val d = c.mul(c)
                    val e = d.add(a)
                    val h = e.add(b)
                    val i = h.relu()
                    return@BOp i.logSoftmax()
                },
            )
        }

        repeat(100) {
            val rank = Random.nextInt(1, 5)
            val shape = IntArray(rank) { Random.nextInt(1, 10) }
            val size = shape.fold(1, Int::times)
            printMessage(shape.joinToString(" X ", "A: "))

            val fa = FloatArray(size) { randomFloat() }
            val fb = FloatArray(size) { randomFloat(excludeZero = true) }

            val ta = Tensor.create(data = fa, shape = shape, requiresGrad = true)
            val tb = Tensor.create(data = fb, shape = shape, requiresGrad = true)

            val da = manager.create(data = fa, shape = shape, requiresGrad = true)
            val db = manager.create(data = fb, shape = shape, requiresGrad = true)

            assertOpResEqual(BOp(da, db, NDArray::add), BOp(ta, tb, Tensor::add))
            assertOpResEqual(BOp(da, db, NDArray::mul), BOp(ta, tb, Tensor::mul))
            assertOpResEqual(UOp(da, NDArray::relu), UOp(ta, Tensor::relu))
            assertOpResEqual(UOp(da, NDArray::sum), UOp(ta, Tensor::sum), 0.1F)

            assertOpResEqual(
                BOp(da, db) { a, b ->
                    val c = a.add(b)
                    val d = c.mul(c)
                    val e = d.add(a)
                    val h = e.add(b)
                    val i = h.relu()
                    return@BOp i
                },
                BOp(ta, tb) { a, b ->
                    val c = a.add(b)
                    val d = c.mul(c)
                    val e = d.add(a)
                    val h = e.add(b)
                    val i = h.relu()
                    return@BOp i
                },
            )
        }

        repeat(100) {
            val rankA = Random.nextInt(1, 5)
            val shapeA = IntArray(rankA) { Random.nextInt(1, 10) }
            val sizeA = shapeA.fold(1, Int::times)

            val rankB = Random.nextInt(1, 5)

            val shapeB =
                IntArray(rankB) { i ->
                        val idxA = rankA - 1 - i

                        val r = Random.nextInt(1, 10)

                        when {
                            idxA < 0 -> r
                            shapeA[idxA] == 1 -> r
                            r < 3 -> 1
                            else -> shapeA[idxA]
                        }
                    }
                    .reversedArray()

            val sizeB = shapeB.fold(1, Int::times)

            printMessage(shapeA.joinToString(" X ", "A: ") + shapeB.joinToString(" X ", ", B: "))

            val fa = FloatArray(sizeA) { randomFloat() }
            val fb = FloatArray(sizeB) { randomFloat(excludeZero = true) }

            val ta = Tensor.create(data = fa, shape = shapeA, requiresGrad = true)
            val tb = Tensor.create(data = fb, shape = shapeB, requiresGrad = true)

            val da = manager.create(data = fa, shape = shapeA, requiresGrad = true)
            val db = manager.create(data = fb, shape = shapeB, requiresGrad = true)

            assertOpResEqual(BOp(da, db, NDArray::add), BOp(ta, tb, Tensor::add))
            assertOpResEqual(BOp(da, db, NDArray::mul), BOp(ta, tb, Tensor::mul))
            assertOpResEqual(UOp(da, NDArray::relu), UOp(ta, Tensor::relu))
            assertOpResEqual(UOp(da, NDArray::sum), UOp(ta, Tensor::sum), 0.1F)

            assertOpResEqual(
                BOp(da, db) { a, b ->
                    val c = a.add(b)
                    val d = c.mul(c)
                    val e = d.add(a)
                    val h = e.add(b)
                    val i = h.relu()
                    return@BOp i
                },
                BOp(ta, tb) { a, b ->
                    val c = a.add(b)
                    val d = c.mul(c)
                    val e = d.add(a)
                    val h = e.add(b)
                    val i = h.relu()
                    return@BOp i
                },
            )
        }
    }

    @Test
    fun testMatmulCorrectness() {
        val fa = floatArrayOf(1F, 2F, 3F, 4F)
        val fb = floatArrayOf(5F, 6F, 7F, 8F)

        val sa = intArrayOf(2, 2)
        val sb = intArrayOf(2, 2)

        val ta = Tensor.create(data = fa, shape = sa, requiresGrad = true)
        val tb = Tensor.create(data = fb, shape = sb, requiresGrad = true)

        val da = manager.create(data = fa, shape = sa, requiresGrad = true)
        val db = manager.create(data = fb, shape = sb, requiresGrad = true)

        val fc = floatArrayOf(1F, 2F, 3F, 4F)
        val sc = intArrayOf(2, 2)

        assertOpResEqual(
            BOp(da, db) { x, W ->
                val m = manager.create(fc, shape = sc, requiresGrad = true)

                val out = x.dot(W)
                val outr = out.relu()
                val outl = outr.logSoftmax()
                val outm = outl.mul(m)
                val outa = outm.add(m)
                outa.sum()
            },
            BOp(ta, tb) { x, W ->
                val m = Tensor.create(fc, shape = sc, requiresGrad = true)

                val out = x.matmul(W)
                val outr = out.relu()
                val outl = outr.logSoftmax()
                val outm = outl.mul(m)
                val outa = outm.add(m)
                outa.sum()
            })
    }

    private fun assertOpResEqual(D: UOp<NDArray>, T: UOp<Tensor>, tol: Float = 0.001F) {
        gc.zeroGradients()
        T.v.zeroGrad()

        val dc = D.fn(D.v)
        val tc = T.fn(T.v)

        tc.backward()
        gc.backward(dc)

        try {
            assertTensorEquals(dc, tc, tol)
            assertTensorEquals(D.v.gradient, T.v.grad!!, tol)
        } catch (t: Throwable) {
            printObjects("=== value ===")

            printObjects(D.v)
            printObjects(T.v)

            printObjects("=== grad ===")

            printObjects(D.v.gradient)
            printObjects(T.v.grad)

            printObjects()
            throw t
        }
    }

    private fun assertOpResEqual(D: BOp<NDArray>, T: BOp<Tensor>, tol: Float = 0.001F) {

        gc.zeroGradients()
        T.a.zeroGrad()
        T.b.zeroGrad()

        val dc = D.fn(D.a, D.b)
        val tc = T.fn(T.a, T.b)

        try {
            tc.backward()
            gc.backward(dc)

            assertTensorEquals(dc, tc, tol)
            assertTensorEquals(D.a.gradient, T.a.grad!!, tol)
            assertTensorEquals(D.b.gradient, T.b.grad!!, tol)
        } catch (t: Throwable) {
            printObjects("=== value ===")

            printObjects(true, "DDD", D.a, D.b)
            printObjects(true, "TTT", T.a, T.b)

            printObjects("=== grad ===")

            printObjects(true, "DDD", D.a.gradient, D.b.gradient)
            printObjects(true, "TTT", T.a.grad, T.b.grad)

            printObjects()
            throw t
        }
    }
}
