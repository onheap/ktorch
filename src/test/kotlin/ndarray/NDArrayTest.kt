package ndarray

import kotlin.random.Random
import kotlin.test.assertEquals
import ndarray.Util.*
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.math.exp
import org.jetbrains.kotlinx.multik.api.math.log
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

typealias MKNDArray = org.jetbrains.kotlinx.multik.ndarray.data.NDArray<Float, DN>

fun MKNDArray.reshape(newShape: IntArray): MKNDArray {
    return when (val newRank = newShape.size) {
        1 -> this.reshape(newShape[0]).asDNArray()
        2 -> this.reshape(newShape[0], newShape[1]).asDNArray()
        3 -> this.reshape(newShape[0], newShape[1], newShape[2]).asDNArray()
        4 -> this.reshape(newShape[0], newShape[1], newShape[2], newShape[3]).asDNArray()
        else ->
            this.reshape(
                    newShape[0],
                    newShape[1],
                    newShape[2],
                    newShape[3],
                    *newShape.slice(4 until newRank).toIntArray())
                .asDNArray()
    }
}

class NDArrayTest {
    @Test
    fun testMk() {
        val a =
            mk.ndarray(
                mk[
                    mk[1, 2, 3],
                    mk[4, 5, 6],
                ])
        val b =
            mk.ndarray(
                mk[
                    mk[1, 2],
                    mk[3, 4],
                    mk[5, 6],
                ])

        a.transpose()

        val c = a.dot(b)
        assertEquals(
            mk.ndarray(
                mk[
                    mk[22, 28],
                    mk[49, 64],
                ]),
            c)
    }

    @Test
    fun testNdArraySumWithAxis() {
        val A = NDArray.of(FloatArray(12) { i -> (i + 1).toFloat() })
        val B = A.reshape(arrOf(4, 3))

        val a = B.sum(0)
        assertNDArraysEqual(a, NDArray.of(arrOf(3), arrOfF(22, 26, 30)))

        val b = B.sum(1)
        assertNDArraysEqual(b, NDArray.of(arrOf(4), arrOfF(6, 15, 24, 33)))

        val C = A.reshape(arrOf(2, 2, 3))

        val c = C.sum(0)
        assertNDArraysEqual(c, NDArray.of(arrOf(2, 3), arrOfF(8, 10, 12, 14, 16, 18)))

        val d = C.sum(1)

        assertNDArraysEqual(d, NDArray.of(arrOf(2, 3), arrOfF(5, 7, 9, 17, 19, 21)))

        val e = C.sum(2)

        assertNDArraysEqual(e, NDArray.of(arrOf(2, 2), arrOfF(6, 15, 24, 33)))

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

            val a = NDArray.of(arrOf(m, n), fa)
            val b = NDArray.of(arrOf(n, p), fb)
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
    fun testArithmeticOperatorsCorrectness() {
        // verify matrix
        repeat(100) {
            val m = Random.nextInt(1, 1024)
            val n = Random.nextInt(1, 1024)

            printMessage("A: $m X $n, B: $m X $n")

            val fa = FloatArray(m * n) { Random.nextInt(50).toFloat() }
            val fb = FloatArray(m * n) { Random.nextInt(50).toFloat() }

            val A = mk.ndarray(fa, m, n)
            val B = mk.ndarray(fb, m, n)
            val AT = A.transpose()
            val BT = B.transpose()

            val C = NDArray(arrOf(m, n), fa)
            val D = NDArray(arrOf(m, n), fb)
            val CT = C.transpose()
            val DT = D.transpose()

            // CC
            assertNDArraysEqual(A.plus(B).asDNArray(), C.add(D))
            assertNDArraysEqual(A.minus(B).asDNArray(), C.sub(D))
            assertNDArraysEqual(A.times(B).asDNArray(), C.mul(D))
            assertNDArraysEqual(A.div(B).asDNArray(), C.div(D))
            assertNDArraysEqual(A.log().asDNArray(), C.log())
            assertNDArraysEqual(A.exp().asDNArray(), C.exp())

            // FF
            assertNDArraysEqual(AT.plus(BT).asDNArray(), CT.add(DT))
            assertNDArraysEqual(AT.minus(BT).asDNArray(), CT.sub(DT))
            assertNDArraysEqual(AT.times(BT).asDNArray(), CT.mul(DT))
            assertNDArraysEqual(AT.div(BT).asDNArray(), CT.div(DT))
            assertNDArraysEqual(AT.log().asDNArray(), CT.log())
            assertNDArraysEqual(AT.exp().asDNArray(), CT.exp())

            val a = mk.ndarray(fa, n, m)
            val b = mk.ndarray(fb, n, m)
            val at = a.transpose()
            val bt = b.transpose()

            val c = NDArray(arrOf(n, m), fa)
            val d = NDArray(arrOf(n, m), fb)
            val ct = c.transpose()
            val dt = d.transpose()

            // CF
            assertNDArraysEqual(A.plus(bt).asDNArray(), C.add(dt))
            assertNDArraysEqual(A.minus(bt).asDNArray(), C.sub(dt))
            assertNDArraysEqual(A.times(bt).asDNArray(), C.mul(dt))
            assertNDArraysEqual(A.div(bt).asDNArray(), C.div(dt))

            // FC
            assertNDArraysEqual(at.plus(B).asDNArray(), ct.add(D))
            assertNDArraysEqual(at.minus(B).asDNArray(), ct.sub(D))
            assertNDArraysEqual(at.times(B).asDNArray(), ct.mul(D))
            assertNDArraysEqual(at.div(B).asDNArray(), ct.div(D))
        }

        // verify random
        repeat(100) {
            val rank = Random.nextInt(1, 5)
            val shape = IntArray(rank) { Random.nextInt(1, 10) }
            val size = shape.fold(1, Int::times)
            printMessage(shape.joinToString(" X ", "A: "))

            val fa = FloatArray(size) { Random.nextInt(50).toFloat() }
            val fb = FloatArray(size) { Random.nextInt(50).toFloat() }

            val A = mk.ndarray(fa.toList(), shape, dimensionOf(rank))
            val B = mk.ndarray(fb.toList(), shape, dimensionOf(rank))
            val AT = A.transpose()
            val BT = B.transpose()

            val C = NDArray(shape, fa)
            val D = NDArray(shape, fb)
            val CT = C.transpose()
            val DT = D.transpose()

            // CC
            assertNDArraysEqual(A.plus(B).asDNArray(), C.add(D))
            assertNDArraysEqual(A.minus(B).asDNArray(), C.sub(D))
            assertNDArraysEqual(A.times(B).asDNArray(), C.mul(D))
            assertNDArraysEqual(A.div(B).asDNArray(), C.div(D))
            assertNDArraysEqual(A.log().asDNArray(), C.log())
            assertNDArraysEqual(A.exp().asDNArray(), C.exp())

            // FF
            assertNDArraysEqual(AT.plus(BT).asDNArray(), CT.add(DT))
            assertNDArraysEqual(AT.minus(BT).asDNArray(), CT.sub(DT))
            assertNDArraysEqual(AT.times(BT).asDNArray(), CT.mul(DT))
            assertNDArraysEqual(AT.div(BT).asDNArray(), CT.div(DT))
            assertNDArraysEqual(AT.log().asDNArray(), CT.log())
            assertNDArraysEqual(AT.exp().asDNArray(), CT.exp())

            val reversedShape = shape.reversedArray()
            val a = mk.ndarray(fa.toList(), reversedShape, dimensionOf(rank))
            val b = mk.ndarray(fb.toList(), reversedShape, dimensionOf(rank))
            val at = a.transpose()
            val bt = b.transpose()

            val c = NDArray(reversedShape, fa)
            val d = NDArray(reversedShape, fb)
            val ct = c.transpose()
            val dt = d.transpose()

            // CF
            assertNDArraysEqual(A.plus(bt).asDNArray(), C.add(dt))
            assertNDArraysEqual(A.minus(bt).asDNArray(), C.sub(dt))
            assertNDArraysEqual(A.times(bt).asDNArray(), C.mul(dt))
            assertNDArraysEqual(A.div(bt).asDNArray(), C.div(dt))

            // FC
            assertNDArraysEqual(at.plus(B).asDNArray(), ct.add(D))
            assertNDArraysEqual(at.minus(B).asDNArray(), ct.sub(D))
            assertNDArraysEqual(at.times(B).asDNArray(), ct.mul(D))
            assertNDArraysEqual(at.div(B).asDNArray(), ct.div(D))
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

            val size = m * n
            val i = randomDivisibleBy(size)
            val j = size / i

            val A = mk.ndarray(f, m, n)
            val B = NDArray(intArrayOf(m, n), f)

            val AT = A.transpose()
            val BT = B.transpose()

            val a = A.reshape(i, j).asDNArray()
            val b = B.reshape(arrOf(i, j))

            assertNDArraysEqual(a, b)
            assertNDArraysEqual(a.transpose(), b.transpose())

            val at = AT.reshape(i, j).asDNArray()
            val bt = BT.reshape(arrOf(i, j))

            assertNDArraysEqual(at, bt)
            assertNDArraysEqual(at.transpose(), bt.transpose())
        }

        // verify random
        repeat(100) {
            val rank = Random.nextInt(1, 5)
            val shape = IntArray(rank) { Random.nextInt(1, 6) }
            val size = shape.fold(1, Int::times)
            printMessage(shape.joinToString(" X ", "A: "))

            val newRank = Random.nextInt(1, 5)
            val newShapeList = mutableListOf<Int>()

            if (size == 1) {
                newShapeList.add(1)
            } else {
                var remainSize = size
                while (remainSize > 1) {
                    if (newShapeList.size == newRank - 1) {
                        newShapeList.add(remainSize)
                        break
                    }
                    val curtSize = randomDivisibleBy(remainSize)
                    newShapeList.add(curtSize)
                    remainSize /= curtSize
                }
            }

            val newShape = newShapeList.toIntArray()
            printMessage(newShape.joinToString(" X ", "B: "))

            val f = FloatArray(size) { Random.nextFloat() }
            val A = mk.ndarray(f.toList(), shape, dimensionOf(rank)).asDNArray()
            val B = NDArray(shape, f)

            val AT = A.transpose()
            val BT = B.transpose()

            val a = A.reshape(newShape).asDNArray()
            val b = B.reshape(newShape)

            assertNDArraysEqual(a, b)
            assertNDArraysEqual(a.transpose(), b.transpose())

            val at = AT.reshape(newShape).asDNArray()
            val bt = BT.reshape(newShape)

            assertNDArraysEqual(at, bt)
            assertNDArraysEqual(at.transpose(), bt.transpose())
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
            var temp = i
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
            var temp = i
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
            var temp = i
            for (j in len - 1 downTo 0) {
                indexes[j] = temp % shape[j]
                temp /= shape[j]
            }

            assertEquals(a.get(indexes), b.get(indexes), 0.001F, message)
        }
    }
}
