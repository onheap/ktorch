package core.tensor

import ai.djl.ndarray.NDArray as DJLNDArray
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.training.GradientCollector
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import tools.*

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

    @Test fun testDJL() {}

    @Test
    fun test() {
        val fa = floatArrayOf(1F, 2F, 3F, 4F)
        val fb = floatArrayOf(5F, 6F, 7F, 8F)
        val sa = intArrayOf(2, 2)
        val sb = intArrayOf(2, 2)

        val ta = Tensor.create(data = fa, shape = sa, requiresGrad = true)
        val tb = Tensor.create(data = fb, shape = sb, requiresGrad = true)

        val da = manager.create(data = fa, shape = sa, requiresGrad = true)
        val db = manager.create(data = fb, shape = sb, requiresGrad = true)

        assertOpResEqual(BOp(da, db, NDArray::add), BOp(ta, tb, Tensor::plus))
        assertOpResEqual(BOp(da, db, NDArray::mul), BOp(ta, tb, Tensor::times))
        assertOpResEqual(BOp(da, db, NDArray::mul), BOp(ta, tb, Tensor::times))

        //        assertOpResEqual(UOp(da, NDArray::log), UOp(ta, Tensor::log))

        println(ta.logSoftmax())
        println(da.logSoftmax(1))
    }

    private fun assertOpResEqual(D: UOp<DJLNDArray>, T: UOp<Tensor>) {
        gc.zeroGradients()
        T.v.zeroGrad()

        val dc = D.fn(D.v)
        val tc = T.fn(T.v)

        assertNDArraysEqual(dc, tc)

        tc.backward()
        gc.backward(dc)

        assertNDArraysEqual(D.v.gradient, T.v.grad!!)
    }

    private fun assertOpResEqual(D: BOp<DJLNDArray>, T: BOp<Tensor>) {
        gc.zeroGradients()
        T.a.zeroGrad()
        T.b.zeroGrad()

        val dc = D.fn(D.a, D.b)
        val tc = T.fn(T.a, T.b)

        assertNDArraysEqual(dc, tc)

        tc.backward()
        gc.backward(dc)

        assertNDArraysEqual(D.a.gradient, T.a.grad!!)
        assertNDArraysEqual(D.b.gradient, T.b.grad!!)
    }
}
