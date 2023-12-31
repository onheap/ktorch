package ndarray

import ai.djl.ndarray.NDManager
import kotlin.random.Random
import ndarray.Util.*
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import tools.*

class NDArrayTest {

    private lateinit var manager: NDManager

    @BeforeEach
    fun setUp() {
        manager = NDManager.newBaseManager()
    }

    @AfterEach
    fun tearDown() {
        manager.close()
    }

    @Test
    fun test() {
        val a = NDArray.of(arrOfF(1))

        val b = a.sum(0)

        printObjects(b)
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

            val A = manager.create(fa, m, n)
            val B = manager.create(fb, n, p)
            val C = A.dot(B)

            val a = NDArray.of(arrOf(m, n), fa)
            val b = NDArray.of(arrOf(n, p), fb)
            val c = a.matmul(b)

            assertNDArrayEquals(C, c)
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

            val A = manager.create(f, m, n).transpose()
            val a = NDArray(intArrayOf(m, n), f).transpose()

            assertNDArrayEquals(A, a)
        }

        // verify random
        repeat(100) {
            val rank = Random.nextInt(1, 5)
            val shape = IntArray(rank) { Random.nextInt(1, 6) }
            val size = shape.fold(1, Int::times)
            printMessage(shape.joinToString(" X ", "A: "))

            val f = FloatArray(size) { randomFloat() }
            val A = manager.create(f, shape).transpose()
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

            val fa = FloatArray(m * n) { randomFloat() }
            val fb = FloatArray(n * p) { randomFloat() }

            val A = manager.create(fa, m, n)
            val B = manager.create(fb, p, n).transpose()
            val C = A.dot(B)

            val a = NDArray(intArrayOf(m, n), fa)
            val b = NDArray(intArrayOf(p, n), fb).transpose()
            val c = a.matmul(b)

            assertNDArrayEquals(C, c)
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

            val A = manager.create(fa, n, m).transpose()
            val B = manager.create(fb, n, p)
            val C = A.dot(B)

            val a = NDArray(intArrayOf(n, m), fa).transpose()
            val b = NDArray(intArrayOf(n, p), fb)
            val c = a.matmul(b)

            assertNDArrayEquals(C, c)
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

            val A = manager.create(fa, n, m).transpose()
            val B = manager.create(fb, p, n).transpose()
            val C = A.dot(B)

            val a = NDArray(intArrayOf(n, m), fa).transpose()
            val b = NDArray(intArrayOf(p, n), fb).transpose()
            val c = a.matmul(b)

            assertNDArrayEquals(C, c)
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

            val A = manager.create(f, m, n)
            val B = NDArray(arrOf(m, n), f)

            assertNDArrayEquals(A.sum(), B.sum())
            assertNDArrayEquals(A.transpose().sum(), B.transpose().sum())

            assertNDArrayEquals(A.max(), B.max())
            assertNDArrayEquals(A.transpose().max(), B.transpose().max())
        }

        // verify random
        repeat(100) {
            val rank = Random.nextInt(1, 5)
            val shape = IntArray(rank) { Random.nextInt(1, 6) }
            val size = shape.fold(1, Int::times)
            printMessage(shape.joinToString(" X ", "A: "))

            val f = FloatArray(size) { randomIntFloat() }
            val A = manager.create(f, shape)
            val B = NDArray(shape, f)

            assertNDArrayEquals(A.sum(), B.sum())
            assertNDArrayEquals(A.transpose().sum(), B.transpose().sum())

            assertNDArrayEquals(A.max(), B.max())
            assertNDArrayEquals(A.transpose().max(), B.transpose().max())
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

            val A = manager.create(f, m, n)
            val B = NDArray(arrOf(m, n), f)

            for (i in 0 until 2) {
                assertNDArrayEquals(A.sum(arrOf(i)), B.sum(i))
                assertNDArrayEquals(A.max(arrOf(i)), B.max(i))

                assertNDArrayEquals(A.sum(arrOf(i), true), B.sum(i, true))
                assertNDArrayEquals(A.max(arrOf(i), true), B.max(i, true))
            }

            val C = A.transpose()
            val D = B.transpose()
            for (i in 0 until 2) {
                assertNDArrayEquals(C.sum(arrOf(i)), D.sum(i))
                assertNDArrayEquals(C.max(arrOf(i)), D.max(i))

                assertNDArrayEquals(C.sum(arrOf(i), true), D.sum(i, true))
                assertNDArrayEquals(C.max(arrOf(i), true), D.max(i, true))
            }
        }

        // verify random
        repeat(100) {
            val rank = Random.nextInt(1, 5)
            val shape = IntArray(rank) { Random.nextInt(1, 6) }
            val size = shape.fold(1, Int::times)
            printMessage(shape.joinToString(" X ", "A: "))

            val f = FloatArray(size) { randomIntFloat() }

            val A = manager.create(f, shape)
            val B = NDArray(shape, f)

            for (i in 0 until A.shape.dimension()) {
                assertNDArrayEquals(A.sum(arrOf(i)), B.sum(i))
                assertNDArrayEquals(A.max(arrOf(i)), B.max(i))

                assertNDArrayEquals(A.sum(arrOf(i), true), B.sum(i, true))
                assertNDArrayEquals(A.max(arrOf(i), true), B.max(i, true))
            }

            val C = A.transpose()
            val D = B.transpose()

            for (i in 0 until C.shape.dimension()) {
                assertNDArrayEquals(C.sum(arrOf(i)), D.sum(i))
                assertNDArrayEquals(C.max(arrOf(i)), D.max(i))

                assertNDArrayEquals(C.sum(arrOf(i), true), D.sum(i, true))
                assertNDArrayEquals(C.max(arrOf(i), true), D.max(i, true))
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

            val A = manager.create(fa, m, n)
            val B = manager.create(fb, m, n)
            val AT = A.transpose()
            val BT = B.transpose()

            val C = NDArray(arrOf(m, n), fa)
            val D = NDArray(arrOf(m, n), fb)
            val CT = C.transpose()
            val DT = D.transpose()

            // CC
            assertNDArrayEquals(A.add(B), C.add(D))
            assertNDArrayEquals(A.add(B), C.addNew(D))
            assertNDArrayEquals(A.sub(B), C.sub(D))
            assertNDArrayEquals(A.mul(B), C.mul(D))
            assertNDArrayEquals(A.div(B), C.div(D))
            assertNDArrayEquals(A.maximum(B), C.maximum(D))
            assertNDArrayEquals(A.minimum(B), C.minimum(D))
            assertNDArrayEquals(A.log(), C.log())
            assertNDArrayEquals(A.exp(), C.exp())

            // FF
            assertNDArrayEquals(AT.add(BT), CT.add(DT))
            assertNDArrayEquals(AT.add(BT), CT.addNew(DT))
            assertNDArrayEquals(AT.sub(BT), CT.sub(DT))
            assertNDArrayEquals(AT.mul(BT), CT.mul(DT))
            assertNDArrayEquals(AT.div(BT), CT.div(DT))
            assertNDArrayEquals(AT.maximum(BT), CT.maximum(DT))
            assertNDArrayEquals(AT.minimum(BT), CT.minimum(DT))
            assertNDArrayEquals(AT.log(), CT.log())
            assertNDArrayEquals(AT.exp(), CT.exp())

            val a = manager.create(fa, n, m)
            val b = manager.create(fb, n, m)
            val at = a.transpose()
            val bt = b.transpose()

            val c = NDArray(arrOf(n, m), fa)
            val d = NDArray(arrOf(n, m), fb)
            val ct = c.transpose()
            val dt = d.transpose()

            // CF
            assertNDArrayEquals(A.add(bt), C.add(dt))
            assertNDArrayEquals(A.add(bt), C.addNew(dt))
            assertNDArrayEquals(A.sub(bt), C.sub(dt))
            assertNDArrayEquals(A.mul(bt), C.mul(dt))
            assertNDArrayEquals(A.div(bt), C.div(dt))
            assertNDArrayEquals(A.maximum(bt), C.maximum(dt))
            assertNDArrayEquals(A.minimum(bt), C.minimum(dt))

            // FC
            assertNDArrayEquals(at.add(B), ct.add(D))
            assertNDArrayEquals(at.add(B), ct.addNew(D))
            assertNDArrayEquals(at.sub(B), ct.sub(D))
            assertNDArrayEquals(at.mul(B), ct.mul(D))
            assertNDArrayEquals(at.div(B), ct.div(D))
            assertNDArrayEquals(at.maximum(B), ct.maximum(D))
            assertNDArrayEquals(at.minimum(B), ct.minimum(D))
        }

        // verify random
        repeat(100) {
            val rank = Random.nextInt(1, 5)
            val shape = IntArray(rank) { Random.nextInt(1, 10) }
            val size = shape.fold(1, Int::times)
            printMessage(shape.joinToString(" X ", "A: "))

            val fa = FloatArray(size) { randomIntFloat(-20..20) }
            val fb = FloatArray(size) { randomIntFloat(excludeZero = true) }

            val A = manager.create(fa, shape)
            val B = manager.create(fb, shape)
            val AT = A.transpose()
            val BT = B.transpose()

            val C = NDArray(shape, fa)
            val D = NDArray(shape, fb)
            val CT = C.transpose()
            val DT = D.transpose()

            // CC
            assertNDArrayEquals(A.add(B), C.add(D))
            assertNDArrayEquals(A.sub(B), C.sub(D))
            assertNDArrayEquals(A.mul(B), C.mul(D))
            assertNDArrayEquals(A.div(B), C.div(D))
            assertNDArrayEquals(A.maximum(B), C.maximum(D))
            assertNDArrayEquals(A.minimum(B), C.minimum(D))
            assertNDArrayEquals(A.log(), C.log())
            assertNDArrayEquals(A.exp(), C.exp())

            // FF
            assertNDArrayEquals(AT.add(BT), CT.add(DT))
            assertNDArrayEquals(AT.sub(BT), CT.sub(DT))
            assertNDArrayEquals(AT.mul(BT), CT.mul(DT))
            assertNDArrayEquals(AT.div(BT), CT.div(DT))
            assertNDArrayEquals(AT.maximum(BT), CT.maximum(DT))
            assertNDArrayEquals(AT.minimum(BT), CT.minimum(DT))
            assertNDArrayEquals(AT.log(), CT.log())
            assertNDArrayEquals(AT.exp(), CT.exp())

            val reversedShape = shape.reversedArray()
            val a = manager.create(fa, reversedShape)
            val b = manager.create(fb, reversedShape)
            val at = a.transpose()
            val bt = b.transpose()

            val c = NDArray(reversedShape, fa)
            val d = NDArray(reversedShape, fb)
            val ct = c.transpose()
            val dt = d.transpose()

            // CF
            assertNDArrayEquals(A.add(bt), C.add(dt))
            assertNDArrayEquals(A.sub(bt), C.sub(dt))
            assertNDArrayEquals(A.mul(bt), C.mul(dt))
            assertNDArrayEquals(A.div(bt), C.div(dt))
            assertNDArrayEquals(A.maximum(bt), C.maximum(dt))
            assertNDArrayEquals(A.minimum(bt), C.minimum(dt))

            // FC
            assertNDArrayEquals(at.add(B), ct.add(D))
            assertNDArrayEquals(at.sub(B), ct.sub(D))
            assertNDArrayEquals(at.mul(B), ct.mul(D))
            assertNDArrayEquals(at.div(B), ct.div(D))
            assertNDArrayEquals(at.maximum(B), ct.maximum(D))
            assertNDArrayEquals(at.minimum(B), ct.minimum(D))
        }

        // verify broadcast
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

            val fa = FloatArray(sizeA) { randomIntFloat(-20..20) }
            val fb = FloatArray(sizeB) { randomIntFloat(excludeZero = true) }

            val A = manager.create(fa, shapeA)
            val B = manager.create(fb, shapeB)
            val AT = A.transpose()
            val BT = B.transpose()

            val C = NDArray(shapeA, fa)
            val D = NDArray(shapeB, fb)
            val CT = C.transpose()
            val DT = D.transpose()

            // CC
            assertNDArrayEquals(A.add(B), C.add(D))
            assertNDArrayEquals(A.sub(B), C.sub(D))
            assertNDArrayEquals(A.mul(B), C.mul(D))
            assertNDArrayEquals(A.div(B), C.div(D))
            assertNDArrayEquals(A.maximum(B), C.maximum(D))
            assertNDArrayEquals(A.minimum(B), C.minimum(D))

            val reversedShapeA = shapeA.reversedArray()
            val reversedShapeB = shapeB.reversedArray()
            val a = manager.create(fa, reversedShapeA)
            val b = manager.create(fb, reversedShapeB)
            val at = a.transpose()
            val bt = b.transpose()

            val c = NDArray(reversedShapeA, fa)
            val d = NDArray(reversedShapeB, fb)
            val ct = c.transpose()
            val dt = d.transpose()

            // CF
            assertNDArrayEquals(A.add(bt), C.add(dt))
            assertNDArrayEquals(A.sub(bt), C.sub(dt))
            assertNDArrayEquals(A.mul(bt), C.mul(dt))
            assertNDArrayEquals(A.div(bt), C.div(dt))
            assertNDArrayEquals(A.maximum(bt), C.maximum(dt))
            assertNDArrayEquals(A.minimum(bt), C.minimum(dt))
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

            val A = manager.create(f, m, n)
            val B = NDArray(intArrayOf(m, n), f)

            val AT = A.transpose()
            val BT = B.transpose()

            val a = A.reshape(i.toLong(), j.toLong())
            val b = B.reshape(i, j)

            assertNDArrayEquals(a, b)
            assertNDArrayEquals(a.transpose(), b.transpose())

            val at = AT.reshape(i.toLong(), j.toLong())
            val bt = BT.reshape(i, j)

            assertNDArrayEquals(at, bt)
            assertNDArrayEquals(at.transpose(), bt.transpose())
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
            val A = manager.create(f, shape)
            val B = NDArray(shape, f)

            val AT = A.transpose()
            val BT = B.transpose()

            val a = A.reshape(*newShape.toLongArray())
            val b = B.reshape(*newShape)

            assertNDArrayEquals(a, b)
            assertNDArrayEquals(a.transpose(), b.transpose())

            val at = AT.reshape(*newShape.toLongArray())
            val bt = BT.reshape(*newShape)

            assertNDArrayEquals(at, bt)
            assertNDArrayEquals(at.transpose(), bt.transpose())
        }
    }
}
