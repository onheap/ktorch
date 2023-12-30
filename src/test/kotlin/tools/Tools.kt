package tools

import ai.djl.ndarray.NDArray as DJLNDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import core.tensor.Tensor
import kotlin.random.Random
import kotlin.reflect.KFunction1
import kotlin.reflect.KFunction2
import kotlin.test.assertEquals
import org.junit.jupiter.api.Assertions.assertArrayEquals

data class BOp<T>(val a: T, val b: T, val fn: KFunction2<T, T, T>)

data class UOp<T>(val v: T, val fn: KFunction1<T, T>)

fun NDManager.create(
    data: FloatArray,
    shape: IntArray = intArrayOf(data.size),
    requiresGrad: Boolean = false
): DJLNDArray {
    return this.create(data, Shape(shape.map(Int::toLong))).also {
        it.setRequiresGradient(requiresGrad)
    }
}

fun IntArray.toLongArray() = this.map(Int::toLong).toLongArray()

fun LongArray.toIntArray() = this.map(Long::toInt).toIntArray()

fun assertNDArraysEqual(a: DJLNDArray, b: Tensor, message: String = "") {
    assertArrayEquals(a.shape.shape, b.shape().toLongArray())

    val shape = a.shape.shape
    val len = shape.size
    val indexes = LongArray(len)
    val totalSize = shape.fold(1, Long::times)

    for (i in 0 until totalSize) {
        var temp = i
        for (j in len - 1 downTo 0) {
            indexes[j] = temp % shape[j]
            temp /= shape[j]
        }

        assertEquals(a.getFloat(*indexes), b.get(*indexes.toIntArray()), 0.001F, message)
    }
}

fun printMessage(message: String?) {
    println(message)
}

private fun printObjects(vararg objs: Any?) {
    println(objs.joinToString(" "))
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
