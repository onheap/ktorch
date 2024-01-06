package tools

import kotlin.math.sqrt
import kotlin.random.Random

object RandomUtil {

    fun uniform(size: Int, rand: Random = Random.Default): FloatArray {
        return FloatArray(size) { rand.nextFloat() * 2 - 1 }
    }

    fun xavierInitArray(size: Int, rand: Random = Random.Default): FloatArray {
        return FloatArray(size) { (rand.nextFloat() * 2 - 1) / sqrt(size.toFloat()) }
    }

    fun randInt(size: Int, range: IntRange, rand: Random = Random.Default): IntArray {
        return IntArray(size) { rand.nextInt(range.first, range.last) }
    }
}
