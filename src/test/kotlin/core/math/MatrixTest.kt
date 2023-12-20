package core.math

import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.junit.jupiter.api.Test
import playground.NDArray
import kotlin.random.Random
import kotlin.test.assertEquals

class MatrixTest {

    @Test
    fun testMk() {
        val a = mk.ndarray(
            mk[
                mk[1, 2, 3],
                mk[4, 5, 6],
            ]
        )
        val b = mk.ndarray(
            mk[
                mk[1, 2],
                mk[3, 4],
                mk[5, 6],
            ]
        )


        a.transpose()

        val c = a.dot(b)
        assertEquals(
            mk.ndarray(
                mk[
                    mk[22, 28],
                    mk[49, 64],
                ]
            ),
            c
        )
    }


    @Test
    fun testMatMulCorrectness() {
        repeat(100) {
            val m = Random.nextInt(2, 1024)
            val n = Random.nextInt(2, 1024)
            val p = Random.nextInt(2, 1024)

            println("A: $m X $n, B: $n X $p")

            val fa = FloatArray(m * n) { Random.nextFloat() }
            val fb = FloatArray(n * p) { Random.nextFloat() }

            val A = mk.ndarray(fa, m, n)
            val B = mk.ndarray(fb, n, p)
            val C = A.dot(B).data.getFloatArray()


            val a = NDArray(intArrayOf(m, n), fa)
            val b = NDArray(intArrayOf(n, p), fb)

            val c1 = a.matMulSimple(b).data
            val c2 = a.matMulVector(b).data
            val c3 = a.matMulVectorConcurrent(b).data

            for (i in 0 until m * p) {
                assertEquals(C[i], c1[i], 0.001f, "Error in matMulSimple")
                assertEquals(C[i], c2[i], 0.001f, "Error in matMulVector")
                assertEquals(C[i], c3[i], 0.001f, "Error in matMulVectorConcurrent")
            }
        }
    }



}