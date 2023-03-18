import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.rand
import kotlin.system.measureNanoTime

fun main(args: Array<String>) {
    calFlops()
}


fun calFlops() {
    fun fmt(double: Double): String = String.format("%.2f", double)

    val N = 1 shl 14  // 16384

    // N^2
    val A = mk.rand<Float>(N, N)
    // N^2
    val B = mk.rand<Float>(N, N)

    // N^2 output with 2N compute each
    // float operation
    val flop = N * N * 2L * N
    println("${fmt(flop / 1e9)} GFLOP")


    val nanos = measureNanoTime {
        A dot B
    }.toDouble()


    println("${fmt(flop / nanos)} GFLOP/s")
}