package core.benchmark

import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.rand
import playground.NDArray
import kotlin.random.Random
import kotlin.system.measureNanoTime

class Benchmark {
//    val N = 1024
    val N = 4096
//    val N = 8192
//    val N = 16384
//    val N = 25000

    private fun fmt(double: Double): String = String.format("%.2f", double)

    private fun getFLOP(): Long = N * N * 2L * N

    fun calFlopsMultik() {
        // N^2
        val A = mk.rand<Float>(N, N)
        // N^2
        val B = mk.rand<Float>(N, N)

        // N^2 output with 2N compute each
        // float operation
        val flop = getFLOP()
        println("== Multik ==")
        println("${fmt(flop / 1e9)} GFLOP")


        val nanos = measureNanoTime {
            A dot B
        }.toDouble()


        println("${fmt(flop / nanos)} GFLOP/s")
    }

    fun calFlopsNDArraySimple() {
        val A = NDArray(intArrayOf(N, N), FloatArray(N * N) { Random.nextFloat() })
        val B = NDArray(intArrayOf(N, N), FloatArray(N * N) { Random.nextFloat() })

        val flop = getFLOP()
        println("== NDArraySimple ==")
        println("${fmt(flop / 1e9)} GFLOP")


        val nanos = measureNanoTime {
            A.matMulSimple(B)
        }.toDouble()

        println("${fmt(flop / nanos)} GFLOP/s")
    }

    fun calFlopsNDArrayVector() {
        val A = NDArray(intArrayOf(N, N), FloatArray(N * N) { Random.nextFloat() })
        val B = NDArray(intArrayOf(N, N), FloatArray(N * N) { Random.nextFloat() })

        val flop = getFLOP()
        println("== NDArrayVector ==")
        println("${fmt(flop / 1e9)} GFLOP")


        val nanos = measureNanoTime {
            A.matMulVector(B)
        }.toDouble()

        println("${fmt(flop / nanos)} GFLOP/s")
    }

    fun calFlopsNDArrayVectorConcurrent() {
        val A = NDArray(intArrayOf(N, N), FloatArray(N * N) { Random.nextFloat() })
        val B = NDArray(intArrayOf(N, N), FloatArray(N * N) { Random.nextFloat() })

        val flop = getFLOP()
        println("== NDArrayVector ==")
        println("${fmt(flop / 1e9)} GFLOP")


        val nanos = measureNanoTime {
            A.matMulVectorConcurrent(B)
        }.toDouble()

        println("${fmt(flop / nanos)} GFLOP/s")
    }
}