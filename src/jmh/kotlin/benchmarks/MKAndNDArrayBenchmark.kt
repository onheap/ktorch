package benchmarks

import java.util.concurrent.TimeUnit
import jdk.incubator.vector.*
import kotlin.random.Random
import ndarray.NDArray
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.openjdk.jmh.annotations.*

typealias MKNDArray = org.jetbrains.kotlinx.multik.ndarray.data.NDArray<Float, D2>

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgsPrepend = ["--add-modules=jdk.incubator.vector", "-XX:-TieredCompilation"])
open class MKAndNDArrayBenchmark {
    @Param("1024") private var size = 0

    private lateinit var A: FloatArray
    private lateinit var B: FloatArray

    private lateinit var MA: MKNDArray
    private lateinit var MB: MKNDArray

    private lateinit var NA: NDArray
    private lateinit var NB: NDArray

    @Setup
    fun setup() {
        A = FloatArray(size * size) { Random.nextFloat() }
        B = FloatArray(size * size) { Random.nextFloat() }

        MA = mk.ndarray(A, size, size)
        MB = mk.ndarray(B, size, size)

        NA = NDArray.of(intArrayOf(size, size), A)
        NB = NDArray.of(intArrayOf(size, size), B)
    }

    @Benchmark
    fun MK_Matmul() {
        MA dot MB
    }

    @Benchmark
    fun MDArray_Matmul() {
        NA.matmul(NB)
    }
}
