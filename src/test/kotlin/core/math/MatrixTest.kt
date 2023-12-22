package core.math

import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.*
import playground.NDArray
import kotlin.random.Random
import kotlin.test.assertEquals

typealias MKNDArray = org.jetbrains.kotlinx.multik.ndarray.data.NDArray<Float, DN>

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
    fun testMatMulCCCorrectness() {
        repeat(100) {
            val m = Random.nextInt(1, 1024)
            val n = Random.nextInt(1, 1024)
            val p = Random.nextInt(1, 1024)

            printMessage("A: $m X $n, B: $n X $p")

            val fa = FloatArray(m * n) { Random.nextFloat() }
            val fb = FloatArray(n * p) { Random.nextFloat() }

            val A = mk.ndarray(fa, m, n)
            val B = mk.ndarray(fb, n, p)
            val C = A.dot(B)


            val a = NDArray(intArrayOf(m, n), fa)
            val b = NDArray(intArrayOf(n, p), fb)
            val c = a.matmul(b)

            assertNDArrayEquals(C.asDNArray(), c)
        }
    }

    @Test
    fun testTransposeCorrectness() {
        // verify matrix
        repeat(100) {
            val m = Random.nextInt(1, 1024)
            val n = Random.nextInt(1, 1024)


            printMessage("Matrix: $m X $n")

            val f = FloatArray(m * n) { Random.nextFloat() }

            val A = mk.ndarray(f, m, n).transpose()
            val a = NDArray(intArrayOf(m, n), f).transpose()

            assertNDArrayEquals(A.asDNArray(), a)
        }


        // verify random
        repeat(100) {
            val rank = Random.nextInt(1, 5)
            val shape = IntArray(rank) { Random.nextInt(1, 6) }
            val size = shape.fold(1, Int::times)
            printMessage(shape.joinToString(" X ", "A: "))

            val f = FloatArray(size) { Random.nextFloat() }
            val A = mk.ndarray(f.toList(), shape, dimensionOf(rank)).asDNArray().transpose()
            val a = NDArray(shape, f).transpose()

            assertNDArrayEquals(A, a)
        }
    }

    @Test
    fun testMatMulCFCorrectness() {
        repeat(100) {
            val m = Random.nextInt(1, 1024)
            val n = Random.nextInt(1, 1024)
            val p = Random.nextInt(1, 1024)

            printMessage("A: $m X $n, B: $n X $p")

            val fa = FloatArray(m * n) { Random.nextFloat() }
            val fb = FloatArray(n * p) { Random.nextFloat() }

            val A = mk.ndarray(fa, m, n)
            val B = mk.ndarray(fb, p, n).transpose()
            val C = A.dot(B)


            val a = NDArray(intArrayOf(m, n), fa)
            val b = NDArray(intArrayOf(p, n), fb).transpose()
            val c = a.matmul(b)

            assertNDArrayEquals(C.asDNArray(), c)
        }
    }

    @Test
    fun testMatMulFCCorrectness() {
        repeat(100) {
            val m = Random.nextInt(1, 1024)
            val n = Random.nextInt(1, 1024)
            val p = Random.nextInt(1, 1024)

            printMessage("A: $m X $n, B: $n X $p")

            val fa = FloatArray(m * n) { Random.nextFloat() }
            val fb = FloatArray(n * p) { Random.nextFloat() }

            val A = mk.ndarray(fa, n, m).transpose()
            val B = mk.ndarray(fb, n, p)
            val C = A.dot(B)


            val a = NDArray(intArrayOf(n, m), fa).transpose()
            val b = NDArray(intArrayOf(n, p), fb)
            val c = a.matmul(b)

            assertNDArrayEquals(C.asDNArray(), c)
        }
    }


    @Test
    fun testMatMulFFCorrectness() {
        repeat(100) {
            val m = Random.nextInt(1, 1024)
            val n = Random.nextInt(1, 1024)
            val p = Random.nextInt(1, 1024)

            printMessage("A: $m X $n, B: $n X $p")

            val fa = FloatArray(m * n) { Random.nextFloat() }
            val fb = FloatArray(n * p) { Random.nextFloat() }

            val A = mk.ndarray(fa, n, m).transpose()
            val B = mk.ndarray(fb, p, n).transpose()
            val C = A.dot(B)


            val a = NDArray(intArrayOf(n, m), fa).transpose()
            val b = NDArray(intArrayOf(p, n), fb).transpose()
            val c = a.matmul(b)

            assertNDArrayEquals(C.asDNArray(), c)
        }
    }

    @Test
    fun testAdd() {
        // verify matrix
        repeat(100) {
            val m = Random.nextInt(1, 1024)
            val n = Random.nextInt(1, 1024)


            printMessage("Matrix: $m X $n")

            val f = FloatArray(m * n) { Random.nextFloat() }

            val A = mk.ndarray(f, m, n).sum()
            val a = NDArray(intArrayOf(m, n), f).sum()

            assertEquals(A, a, 0.1F)
        }


        // verify random
        repeat(100) {
            val rank = Random.nextInt(1, 5)
            val shape = IntArray(rank) { Random.nextInt(1, 6) }
            val size = shape.fold(1, Int::times)
            printMessage(shape.joinToString(" X ", "A: "))

            val f = FloatArray(size) { Random.nextFloat() }
            val A = mk.ndarray(f.toList(), shape, dimensionOf(rank)).asDNArray().transpose().sum()
            val a = NDArray(shape, f).transpose().sum()

            assertEquals(A, a, 0.1F)
        }
    }


    @Test
    fun testReshape() {
        // verify matrix
        repeat(0) {
            val m = Random.nextInt(1, 1024)
            val n = Random.nextInt(1, 1024)


            printMessage("Matrix: $m X $n")

            val f = FloatArray(m * n) { Random.nextFloat() }

            val size = m * n;
            val k = randomDivisibleBy(size)
            val A = mk.ndarray(f, m, n).reshape(k, size / k).asDNArray()
            val a = NDArray(intArrayOf(m, n), f).reshape(intArrayOf(k, size / k))
            assertNDArrayEquals(A, a)
        }


        // verify random
        repeat(100) {
            val rank = Random.nextInt(1, 5)
            val shape = IntArray(rank) { Random.nextInt(1, 6) }
            val size = shape.fold(1, Int::times)
            printMessage(shape.joinToString(" X ", "A: "))

            val newRank = Random.nextInt(1, 5)
            val newShape = mutableListOf<Int>()

            if (size == 1) {
                newShape.add(1)
            } else {
                var remainSize = size;
                while (remainSize > 1) {
                    if (newShape.size == newRank - 1) {
                        newShape.add(remainSize)
                        break
                    }
                    val curtSize = randomDivisibleBy(remainSize)
                    // println("curtSize = $curtSize, remainSize = $remainSize")
                    newShape.add(curtSize)
                    remainSize /= curtSize
                }
            }

            printMessage(newShape.joinToString(" X ", "B: "))


            val f = FloatArray(size) { Random.nextFloat() }
            val A = mk.ndarray(f.toList(), shape, dimensionOf(rank)).asDNArray()
            val a = NDArray(shape, f).transpose().sum()

//            assertEquals(A, a, 0.1F)
        }
    }


    private fun printMessage(message: String?) {
        println(message)
    }

    private fun randomDivisibleBy(v : Int): Int {
        var i = Random.nextInt(1, v + 1)
        while (i == 0 || v % i != 0) {
            i = Random.nextInt(v)
        }
        return i
    }


    private fun assertNDArrayEquals(a: MKNDArray, b: NDArray, message: String? = null) {

        assertArrayEquals(a.shape, b.shape)
        assertArrayEquals(a.strides, b.strides)

        val shape = a.shape
        val len = shape.size
        val indexes = IntArray(len)
        val totalSize = shape.fold(1, Int::times)

        for (i in 0 until totalSize) {
            var temp = i;
            for (j in len - 1 downTo 0) {
                indexes[j] = temp % shape[j]
                temp /= shape[j]
            }

            assertEquals(a.get(indexes), b.get(indexes), 0.001F)
        }
    }
}