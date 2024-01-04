package benchmarks

import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.system.measureNanoTime
import ndarray.NDArray
import ndarray.NDArrays
import org.ejml.simple.SimpleMatrix
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.rand

class MatmulFLOPS(val N: Int) {

    private fun fmt(double: Double): String = String.format("%.2f", double)

    private fun getFLOP(): Long = N * N * 2L * N

    private fun avgMatrixSize(params: Int, layers: Int): Int {
        return sqrt(params.toDouble() / layers).toInt()
    }

    fun info() {
        val flop = getFLOP()
        println("== Info ==")
        println("Matrix size: $N X $N")
        println("Total FLOPs ${fmt(flop / 1e9)} GFLOP")
    }

    fun MK_Matmul() {
        val A = mk.rand<Float>(N, N)
        val B = mk.rand<Float>(N, N)

        val nanos = measureNanoTime { A dot B }.toDouble()

        println("Multik Matmul a @ b ${fmt(getFLOP() / nanos)} GFLOP/s")
    }

    fun EJML_Matmul() {
        val A = SimpleMatrix(N, N, true, *FloatArray(N * N) { Random.nextFloat() })
        val B = SimpleMatrix(N, N, true, *FloatArray(N * N) { Random.nextFloat() })

        val nanos = measureNanoTime { A.mult(B) }.toDouble()

        println("EJML Matmul a @ b ${fmt(getFLOP() / nanos)} GFLOP/s")
    }

    fun NDArray_Matmul() {
        val A = NDArrays.of(intArrayOf(N, N), FloatArray(N * N) { Random.nextFloat() })
        val B = NDArrays.of(intArrayOf(N, N), FloatArray(N * N) { Random.nextFloat() })

        val nanos = measureNanoTime { A.matmul(B) }.toDouble()

        println("NDArray Matmul a @ b ${fmt(getFLOP() / nanos)} GFLOP/s")
    }

    fun MK_Matmul_ABT() {
        val A = mk.rand<Float>(N, N)
        val B = mk.rand<Float>(N, N).transpose()

        val nanos = measureNanoTime { A dot B }.toDouble()

        println("Multik Matmul a @ b.T ${fmt(getFLOP() / nanos)} GFLOP/s")
    }

    fun NDArray_Matmul_ABT() {
        val A = NDArray(intArrayOf(N, N), FloatArray(N * N) { Random.nextFloat() })
        val B = NDArray(intArrayOf(N, N), FloatArray(N * N) { Random.nextFloat() }).transpose()

        val nanos = measureNanoTime { A.matmul(B) }.toDouble()

        println("NDArray Matmul a @ b.T ${fmt(getFLOP() / nanos)} GFLOP/s")
    }

    fun MK_Matmul_ATB() {
        val A = mk.rand<Float>(N, N).transpose()
        val B = mk.rand<Float>(N, N)

        val nanos = measureNanoTime { A dot B }.toDouble()

        println("Multik Matmul a.T @ b ${fmt(getFLOP() / nanos)} GFLOP/s")
    }

    fun NDArray_Matmul_ATB() {
        val A = NDArray(intArrayOf(N, N), FloatArray(N * N) { Random.nextFloat() }).transpose()
        val B = NDArray(intArrayOf(N, N), FloatArray(N * N) { Random.nextFloat() })

        val nanos = measureNanoTime { A.matmul(B) }.toDouble()

        println("NDArray Matmul a.T @ b ${fmt(getFLOP() / nanos)} GFLOP/s")
    }

    fun MK_Matmul_ATBT() {
        val A = mk.rand<Float>(N, N).transpose()
        val B = mk.rand<Float>(N, N).transpose()

        val nanos = measureNanoTime { A dot B }.toDouble()

        println("Multik Matmul a.T @ b.T ${fmt(getFLOP() / nanos)} GFLOP/s")
    }

    fun NDArray_Matmul_ATBT() {
        val A = NDArray(intArrayOf(N, N), FloatArray(N * N) { Random.nextFloat() }).transpose()
        val B = NDArray(intArrayOf(N, N), FloatArray(N * N) { Random.nextFloat() }).transpose()

        val nanos = measureNanoTime { A.matmul(B) }.toDouble()

        println("NDArray Matmul a.T @ b.T ${fmt(getFLOP() / nanos)} GFLOP/s")
    }
}

fun main(args: Array<String>) {
    //    val N = 1024
    //    val N = 4096
    //    val N = 7365
    //    val N = 8192
    //    val N = 16384
    //    val N = 25000

    val benchmark = MatmulFLOPS(4096)
    benchmark.info()

    println("\n== a @ b ==")
    benchmark.MK_Matmul()
    benchmark.NDArray_Matmul()
    benchmark.EJML_Matmul()

    //    println("\n== a @ b.T ==")
    //    benchmark.MK_Matmul_ABT()
    //    benchmark.NDArray_Matmul_ABT()
    //
    //    println("\n== a.T @ b ==")
    //    benchmark.MK_Matmul_ATB()
    //    benchmark.NDArray_Matmul_ATB()
    //
    //    println("\n== a.T @ b.T ==")
    //    benchmark.MK_Matmul_ATBT()
    //    benchmark.NDArray_Matmul_ATBT()
}
