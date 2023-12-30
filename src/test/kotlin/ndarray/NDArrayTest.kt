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
import tools.*

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
    fun testNdArray() {
        val a = NDArray.arange(0, 12).reshape(4, 3)

        val b = NDArray.arange(0, 6).reshape(2, 1, 3)
        println(a)
        println(b)

        println(a.add(b))
    }

    @Test
    fun testMatMulCCCorrectness() {
        repeat(100) {
            val m = Random.nextInt(1, 1024)
            val n = Random.nextInt(1, 1024)
            val p = Random.nextInt(1, 1024)

            printMessage("A: $m X $n, B: $n X $p")

            val fa = FloatArray(m * n) { randomFloat() }
            val fb = FloatArray(n * p) { randomFloat() }

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

            val f = FloatArray(m * n) { randomFloat() }

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

            val f = FloatArray(size) { randomFloat() }
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

            val fa = FloatArray(m * n) { randomFloat() }
            val fb = FloatArray(n * p) { randomFloat() }

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

            val fa = FloatArray(m * n) { randomFloat() }
            val fb = FloatArray(n * p) { randomFloat() }

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

            val fa = FloatArray(m * n) { randomFloat() }
            val fb = FloatArray(n * p) { randomFloat() }

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
    fun testReduceCorrectness() {
        // verify matrix
        repeat(100) {
            val m = Random.nextInt(1, 1024)
            val n = Random.nextInt(1, 1024)

            printMessage("Matrix: $m X $n")

            val f = FloatArray(m * n) { randomIntFloat() }

            val A = mk.ndarray(f, m, n)
            val B = NDArray(arrOf(m, n), f)

            assertEquals(A.sum(), B.sum().asScalar(), 0.1F)
            assertEquals(A.transpose().sum(), B.transpose().sum().asScalar(), 0.1F)

            assertEquals(A.max()!!, B.max().asScalar(), 0.1F)
            assertEquals(A.transpose().max()!!, B.transpose().max().asScalar(), 0.1F)
        }

        // verify random
        repeat(100) {
            val rank = Random.nextInt(1, 5)
            val shape = IntArray(rank) { Random.nextInt(1, 6) }
            val size = shape.fold(1, Int::times)
            printMessage(shape.joinToString(" X ", "A: "))

            val f = FloatArray(size) { randomIntFloat() }
            val A = mk.ndarray(f.toList(), shape, dimensionOf(rank)).asDNArray()
            val B = NDArray(shape, f)

            assertEquals(A.sum(), B.sum().asScalar(), 0.1F)
            assertEquals(A.transpose().sum(), B.transpose().sum().asScalar(), 0.1F)

            assertEquals(A.max()!!, B.max().asScalar(), 0.1F)
            assertEquals(A.transpose().max()!!, B.transpose().max().asScalar(), 0.1F)
        }
    }

    @Test
    fun testReduceAlongDimensionCorrectness() {
        // verify matrix
        repeat(100) {
            val m = Random.nextInt(1, 1024)
            val n = Random.nextInt(1, 1024)

            printMessage("Matrix: $m X $n")

            val f = FloatArray(m * n) { randomFloat() }

            val A = mk.ndarray(f, m, n)
            val B = NDArray(arrOf(m, n), f)

            for (i in 0 until 2) {
                assertNDArraysEqual(mk.math.sumD2(A, i).asDNArray(), B.sum(i))
                assertNDArraysEqual(mk.math.maxD2(A, i).asDNArray(), B.max(i))
            }

            val C = A.transpose()
            val D = B.transpose()
            for (i in 0 until 2) {
                assertNDArraysEqual(mk.math.sumD2(C, i).asDNArray(), D.sum(i))
                assertNDArraysEqual(mk.math.maxD2(C, i).asDNArray(), D.max(i))
            }
        }

        // verify random
        repeat(100) {
            val rank = 3
            val shape = IntArray(rank) { Random.nextInt(1, 3) }
            val size = shape.fold(1, Int::times)
            printMessage(shape.joinToString(" X ", "A: "))

            val f = FloatArray(size) { randomIntFloat(-20..20) }

            val A = mk.ndarray(f, shape[0], shape[1], shape[2])
            val B = NDArray(shape, f)

            for (i in 0..2) {
                assertNDArraysEqual(mk.math.sumD3(A, i).asDNArray(), B.sum(i))
                assertNDArraysEqual(mk.math.maxD3(A, i).asDNArray(), B.max(i))
            }

            val C = A.transpose()
            val D = B.transpose()

            for (i in 0..2) {
                assertNDArraysEqual(mk.math.sumD3(C, i).asDNArray(), D.sum(i))
                assertNDArraysEqual(mk.math.maxD3(C, i).asDNArray(), D.max(i))
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

            val fa = FloatArray(m * n) { randomIntFloat(-20..20) }
            val fb = FloatArray(m * n) { randomIntFloat(excludeZero = true) }

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
            assertNDArraysEqual(A.plus(B).asDNArray(), C.addNew(D))
            assertNDArraysEqual(A.minus(B).asDNArray(), C.sub(D))
            assertNDArraysEqual(A.times(B).asDNArray(), C.mul(D))
            assertNDArraysEqual(A.div(B).asDNArray(), C.div(D))
            assertNDArraysEqual(A.maximum(B).asDNArray(), C.maximum(D))
            assertNDArraysEqual(A.minimum(B).asDNArray(), C.minimum(D))
            assertNDArraysEqual(A.log().asDNArray(), C.log())
            assertNDArraysEqual(A.exp().asDNArray(), C.exp())

            // FF
            assertNDArraysEqual(AT.plus(BT).asDNArray(), CT.add(DT))
            assertNDArraysEqual(AT.plus(BT).asDNArray(), CT.addNew(DT))
            assertNDArraysEqual(AT.minus(BT).asDNArray(), CT.sub(DT))
            assertNDArraysEqual(AT.times(BT).asDNArray(), CT.mul(DT))
            assertNDArraysEqual(AT.div(BT).asDNArray(), CT.div(DT))
            assertNDArraysEqual(AT.maximum(BT).asDNArray(), CT.maximum(DT))
            assertNDArraysEqual(AT.minimum(BT).asDNArray(), CT.minimum(DT))
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
            assertNDArraysEqual(A.plus(bt).asDNArray(), C.addNew(dt))
            assertNDArraysEqual(A.minus(bt).asDNArray(), C.sub(dt))
            assertNDArraysEqual(A.times(bt).asDNArray(), C.mul(dt))
            assertNDArraysEqual(A.div(bt).asDNArray(), C.div(dt))
            assertNDArraysEqual(A.maximum(bt).asDNArray(), C.maximum(dt))
            assertNDArraysEqual(A.minimum(bt).asDNArray(), C.minimum(dt))

            // FC
            assertNDArraysEqual(at.plus(B).asDNArray(), ct.add(D))
            assertNDArraysEqual(at.plus(B).asDNArray(), ct.addNew(D))
            assertNDArraysEqual(at.minus(B).asDNArray(), ct.sub(D))
            assertNDArraysEqual(at.times(B).asDNArray(), ct.mul(D))
            assertNDArraysEqual(at.div(B).asDNArray(), ct.div(D))
            assertNDArraysEqual(at.maximum(B).asDNArray(), ct.maximum(D))
            assertNDArraysEqual(at.minimum(B).asDNArray(), ct.minimum(D))
        }

        // verify random
        repeat(100) {
            val rank = Random.nextInt(1, 5)
            val shape = IntArray(rank) { Random.nextInt(1, 10) }
            val size = shape.fold(1, Int::times)
            printMessage(shape.joinToString(" X ", "A: "))

            val fa = FloatArray(size) { randomIntFloat(-20..20) }
            val fb = FloatArray(size) { randomIntFloat(excludeZero = true) }

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
            assertNDArraysEqual(A.maximum(B).asDNArray(), C.maximum(D))
            assertNDArraysEqual(A.minimum(B).asDNArray(), C.minimum(D))
            assertNDArraysEqual(A.log().asDNArray(), C.log())
            assertNDArraysEqual(A.exp().asDNArray(), C.exp())

            // FF
            assertNDArraysEqual(AT.plus(BT).asDNArray(), CT.add(DT))
            assertNDArraysEqual(AT.minus(BT).asDNArray(), CT.sub(DT))
            assertNDArraysEqual(AT.times(BT).asDNArray(), CT.mul(DT))
            assertNDArraysEqual(AT.div(BT).asDNArray(), CT.div(DT))
            assertNDArraysEqual(AT.maximum(BT).asDNArray(), CT.maximum(DT))
            assertNDArraysEqual(AT.minimum(BT).asDNArray(), CT.minimum(DT))
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
            assertNDArraysEqual(A.maximum(bt).asDNArray(), C.maximum(dt))
            assertNDArraysEqual(A.minimum(bt).asDNArray(), C.minimum(dt))

            // FC
            assertNDArraysEqual(at.plus(B).asDNArray(), ct.add(D))
            assertNDArraysEqual(at.minus(B).asDNArray(), ct.sub(D))
            assertNDArraysEqual(at.times(B).asDNArray(), ct.mul(D))
            assertNDArraysEqual(at.div(B).asDNArray(), ct.div(D))
            assertNDArraysEqual(at.maximum(B).asDNArray(), ct.maximum(D))
            assertNDArraysEqual(at.minimum(B).asDNArray(), ct.minimum(D))
        }
    }

    @Test
    fun testReshapeCorrectness() {
        // verify matrix
        repeat(100) {
            val m = Random.nextInt(1, 1024)
            val n = Random.nextInt(1, 1024)

            printMessage("Matrix: $m X $n")

            val f = FloatArray(m * n) { randomFloat() }

            val size = m * n
            val i = randomDivisibleBy(size)
            val j = size / i

            val A = mk.ndarray(f, m, n)
            val B = NDArray(intArrayOf(m, n), f)

            val AT = A.transpose()
            val BT = B.transpose()

            val a = A.reshape(i, j).asDNArray()
            val b = B.reshape(i, j)

            assertNDArraysEqual(a, b)
            assertNDArraysEqual(a.transpose(), b.transpose())

            val at = AT.reshape(i, j).asDNArray()
            val bt = BT.reshape(i, j)

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

            val f = FloatArray(size) { randomFloat() }
            val A = mk.ndarray(f.toList(), shape, dimensionOf(rank)).asDNArray()
            val B = NDArray(shape, f)

            val AT = A.transpose()
            val BT = B.transpose()

            val a = A.reshape(newShape).asDNArray()
            val b = B.reshape(*newShape)

            assertNDArraysEqual(a, b)
            assertNDArraysEqual(a.transpose(), b.transpose())

            val at = AT.reshape(newShape).asDNArray()
            val bt = BT.reshape(*newShape)

            assertNDArraysEqual(at, bt)
            assertNDArraysEqual(at.transpose(), bt.transpose())
        }
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
