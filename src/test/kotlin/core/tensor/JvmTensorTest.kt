package core.tensor

import ai.djl.ndarray.NDArray as DJLNDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape as DJLShape
import org.junit.jupiter.api.Test

fun IntArray.toDJLShape() = DJLShape(this.map(Int::toLong))

class JvmTensorTest {
    @Test
    fun testDJL() {
        NDManager.newBaseManager().use { manager ->
            val a: DJLNDArray = manager.create(floatArrayOf())

        }
    }

    @Test
    fun test() {
        val fa = floatArrayOf(1F, 2F, 3F, 4F)
        val fb = floatArrayOf(0F, 1F, 2F, 3F)
        val sa = intArrayOf(2, 2)
        val sb = intArrayOf(2, 2)

        val ta = Tensor.create(data = fa, shape = sa, requiresGrad = true)
        val tb = Tensor.create(data = fb, shape = sb, requiresGrad = true)

        val tc = ta * tb
        tc.backward()
        println(tc.grad)
        println(tb.grad)
        println(ta.grad)

        NDManager.newBaseManager().use { manager ->
            val da: DJLNDArray =
                manager.create(fa, sa.toDJLShape()).also { it.setRequiresGradient(true) }
            val db: DJLNDArray =
                manager.create(fb, sb.toDJLShape()).also { it.setRequiresGradient(true) }

            val dc = da.mul(db).also { it.setRequiresGradient(true) }

            val gc = manager.engine.newGradientCollector()

            gc.backward(dc)

            println(dc.gradient)
            println(db.gradient)
            println(da.gradient)
        }
    }
}
