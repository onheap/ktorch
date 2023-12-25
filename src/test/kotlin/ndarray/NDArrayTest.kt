package ndarray

import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.*
import ndarray.Util.*
import kotlin.random.Random
import kotlin.test.assertEquals


typealias MKNDArray = org.jetbrains.kotlinx.multik.ndarray.data.NDArray<Float, DN>

class NDArrayTest {
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
    fun testNdArraySumWithAxis() {
        val A = NDArray.of(FloatArray(12) { i -> (i + 1).toFloat() })
        val B = A.reshape(arrOf(4, 3))

        val a = B.sum(0)
        assertNDArraysEqual(
            a, NDArray.of(arrOf(3), arrOfF(22, 26, 30))
        )

        val b = B.sum(1)
        assertNDArraysEqual(
            b, NDArray.of(arrOf(4), arrOfF(6, 15, 24, 33))
        )

        val C = A.reshape(arrOf(2, 2, 3))

        val c = C.sum(0)
        assertNDArraysEqual(
            c, NDArray.of(arrOf(2, 3), arrOfF(8, 10, 12, 14, 16, 18))
        )

        val d = C.sum(1)

        assertNDArraysEqual(
            d, NDArray.of(arrOf(2, 3), arrOfF(5, 7, 9, 17, 19, 21))
        )

        val e = C.sum(2)

        assertNDArraysEqual(
            e, NDArray.of(arrOf(2, 2), arrOfF(6, 15, 24, 33))
        )

        val D = NDArray.of(arrOfF(16, 7, 17, 1, 7, 3, 8, 7)).reshape(arrOf(2, 2, 2))

        val f = D.sum(0)
        assertNDArraysEqual(f, NDArray.of(arrOf(2, 2), arrOfF(23, 10, 25, 8)))

        val E = NDArray.of(arrOfF(1, 2)).reshape(arrOf(1, 2))
        val g = E.sum(0)
        assertNDArraysEqual(g, NDArray.of(arrOfF(1, 2)))
        val h = E.sum(1)
        assertNDArraysEqual(h, NDArray.of(arrOfF(3)))

        val F = NDArray.of(arrOf(2, 2, 1), arrOfF(16, 9, 13, 16)).transpose()
        val i = F.sum(0)
        assertNDArraysEqual(i, NDArray.of(arrOf(2, 2), arrOfF(16, 13, 9, 16)))
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

            assertNDArraysEqual(C.asDNArray(), c)
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

            assertNDArraysEqual(A.asDNArray(), a)
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

            assertNDArraysEqual(A, a)
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

            assertNDArraysEqual(C.asDNArray(), c)
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

            assertNDArraysEqual(C.asDNArray(), c)
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

            assertNDArraysEqual(C.asDNArray(), c)
        }
    }

    @Test
    fun testSumCorrectness() {
        // verify matrix
        repeat(100) {
            val m = Random.nextInt(1, 1024)
            val n = Random.nextInt(1, 1024)


            printMessage("Matrix: $m X $n")

            val f = FloatArray(m * n) { Random.nextFloat() }

            val A = mk.ndarray(f, m, n).sum()
            val a = NDArray(arrOf(m, n), f).sum().asScalar()

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
            val a = NDArray(shape, f).transpose().sum().asScalar()

            assertEquals(A, a, 0.1F)
        }
    }

    @Test
    fun testSumWithAxisCorrectness() {
        // verify matrix
        repeat(100) {
            val m = Random.nextInt(1, 1024)
            val n = Random.nextInt(1, 1024)


            printMessage("Matrix: $m X $n")

            val f = FloatArray(m * n) { Random.nextFloat() }

            var A = mk.ndarray(f, m, n)
            var B = NDArray(arrOf(m, n), f)


            var a = mk.math.sum<Float, D2, D1>(A, 0).asDNArray()
            var b = B.sum(0)
            assertNDArraysEqual(a, b)

            var c = mk.math.sum<Float, D2, D1>(A, 1).asDNArray()
            var d = B.sum(1)
            assertNDArraysEqual(c, d)


            A = A.transpose()
            B = B.transpose()

            a = mk.math.sum<Float, D2, D1>(A, 0).asDNArray()
            b = B.sum(0)
            assertNDArraysEqual(a, b)

            c = mk.math.sum<Float, D2, D1>(A, 1).asDNArray()
            d = B.sum(1)
            assertNDArraysEqual(c, d)
        }


        // verify random
        repeat(100) {
            val rank = 3
            val shape = IntArray(rank) { Random.nextInt(1, 3) }
            val size = shape.fold(1, Int::times)
            printMessage(shape.joinToString(" X ", "A: "))

            val f = FloatArray(size) { Random.nextInt(20).toFloat() }
            val A = mk.ndarray(f, shape[0], shape[1], shape[2])
            val B = NDArray(shape, f)

            for (i in 0..2) {
                val a = mk.math.sum<Float, D3, D2>(A, i).asDNArray()
                val b = B.sum(i)
                assertNDArraysEqual(a, b)
            }

            val C = A.transpose()
            val D = B.transpose()

            for (i in 0..2) {
                val a = mk.math.sum<Float, D3, D2>(C, i).asDNArray()
                val b = D.sum(i)

                assertNDArraysEqual(a, b)
            }
        }
    }


    @Test
    fun testReshapeCorrectness() {
        // verify matrix
        repeat(100) {
            val m = Random.nextInt(1, 1024)
            val n = Random.nextInt(1, 1024)


            printMessage("Matrix: $m X $n")

            val f = FloatArray(m * n) { Random.nextFloat() }

            val size = m * n;
            val k = randomDivisibleBy(size)
            val A = mk.ndarray(f, m, n).reshape(k, size / k).asDNArray()
            val a = NDArray(intArrayOf(m, n), f).reshape(intArrayOf(k, size / k))
            assertNDArraysEqual(A, a)
        }


        // verify random
        repeat(100) {
            val rank = Random.nextInt(1, 5)
            val shape = IntArray(rank) { Random.nextInt(1, 6) }
            val size = shape.fold(1, Int::times)
            printMessage(shape.joinToString(" X ", "A: "))

            var newRank = Random.nextInt(1, 5)
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

            newRank = newShape.size

            printMessage(newShape.joinToString(" X ", "B: "))

            val f = FloatArray(size) { Random.nextFloat() }
            val A = mk.ndarray(f.toList(), shape, dimensionOf(rank))

            val a = when (newRank) {
                1 -> A.reshape(newShape[0]).asDNArray()
                2 -> A.reshape(newShape[0], newShape[1]).asDNArray()
                3 -> A.reshape(newShape[0], newShape[1], newShape[2]).asDNArray()
                4 -> A.reshape(newShape[0], newShape[1], newShape[2], newShape[3]).asDNArray()
                else -> A.reshape(
                    newShape[0],
                    newShape[1],
                    newShape[2],
                    newShape[3],
                    *newShape.slice(4 until newRank).toIntArray()
                ).asDNArray()
            }


            val b = NDArray(shape, f).transpose().reshape(newShape.toIntArray());

            assertNDArraysEqual(a, b)
        }
    }


    private fun printMessage(message: String?) {
        println(message)
    }

    private fun printObjects(vararg objs: Any) {
        println(objs.joinToString(" "))
    }

    private fun randomDivisibleBy(v: Int): Int {
        var i = Random.nextInt(1, v + 1)
        while (i == 0 || v % i != 0) {
            i = Random.nextInt(v)
        }
        return i
    }


    private fun assertNDArraysEqual(a: MKNDArray, b: NDArray, message: String? = null) {
        assertArrayEquals(a.shape, b.shape)

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

            assertEquals(a.get(indexes), b.get(indexes), 0.001F, message)
        }
    }

    private fun assertNDArraysEqual(a: NDArray, b: NDArray, message: String? = null) {
        assertArrayEquals(a.shape, b.shape)

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

            assertEquals(a.get(indexes), b.get(indexes), 0.001F, message)
        }
    }

    private fun assertNDArraysStrictEqual(a: NDArray, b: NDArray, message: String? = null) {
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

            assertEquals(a.get(indexes), b.get(indexes), 0.001F, message)
        }
    }
}