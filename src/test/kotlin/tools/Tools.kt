package tools

import ai.djl.ndarray.NDArray as DJLNDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import core.tensor.Tensor
import kotlin.random.Random
import kotlin.test.assertEquals
import ndarray.NDArray
import org.junit.jupiter.api.Assertions.assertArrayEquals

data class BOp<T>(val a: T, val b: T, val fn: (T, T) -> T)

data class UOp<T>(val v: T, val fn: (T) -> T)

fun NDManager.create(
    data: FloatArray,
    shape: IntArray = intArrayOf(data.size),
    requiresGrad: Boolean = false
): DJLNDArray {
    return this.create(data, Shape(shape.map(Int::toLong))).also {
        it.setRequiresGradient(requiresGrad)
    }
}

fun NDManager.create(data: FloatArray, a: Int, b: Int): DJLNDArray {
    return this.create(data, Shape(a.toLong(), b.toLong()))
}

fun IntArray.toLongArray() = this.map(Int::toLong).toLongArray()

fun LongArray.toIntArray() = this.map(Long::toInt).toIntArray()

fun assertNDArrayEquals(a: DJLNDArray, b: NDArray, tol: Float = 0.001F, message: String = "") {
    var a = if (a.dataType == DataType.FLOAT32) a else a.toType(DataType.FLOAT32, true)

    if (a.isScalar || b.isScalar) {
        assertEquals(a.isScalar, b.isScalar, message)
        assertEquals(a.getFloat(), b.asScalar(), tol, message)
        return
    }

    assertArrayEquals(a.shape.shape, b.shape.toLongArray())
    assertArrayEquals(a.toFloatArray(), b.toArray(), tol, message)
}

fun assertTensorEquals(a: DJLNDArray, b: Tensor, tol: Float = 0.001F, message: String = "") {
    var a = if (a.dataType == DataType.FLOAT32) a else a.toType(DataType.FLOAT32, true)

    if (a.isScalar || b.isScalar()) {
        assertEquals(a.isScalar, b.isScalar(), message)
        assertEquals(a.getFloat(), b.asScalar(), tol, message)
        return
    }

    assertArrayEquals(a.shape.shape, b.shape().toLongArray())
    assertArrayEquals(a.toFloatArray(), b.toArray(), tol, message)
}

fun printMessage(message: String?) {
    println(message)
}

fun printObjects(vararg objs: Any?) {
    printMessage(objs.joinToString(" "))
}

fun printObjects(vertical: Boolean = false, vararg objs: Any?) {
    if (vertical) {
        printMessage(objs.joinToString("\n"))
    } else {
        printMessage(objs.joinToString(" "))
    }
}

fun randomDivisibleBy(v: Int): Int {
    var i = Random.nextInt(1, v + 1)
    while (i == 0 || v % i != 0) {
        i = Random.nextInt(v)
    }
    return i
}

fun randomFloat(excludeZero: Boolean = false): Float {
    var r = Random.nextFloat() - 0.5F
    while (excludeZero && r == 0F) {
        r = Random.nextFloat() - 0.5F
    }
    return r
}

fun randomIntFloat(range: IntRange = -64 until 64, excludeZero: Boolean = false): Float {
    var r = Random.nextInt(range.first, range.last)
    while (excludeZero && r == 0) {
        r = Random.nextInt(range.first, range.last)
    }
    return r.toFloat()
}
