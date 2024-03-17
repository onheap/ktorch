package benchmarks

import java.util.concurrent.TimeUnit
import kotlin.random.Random
import ndarray.NDArray
import ndarray.NDArrays
import org.openjdk.jmh.annotations.*

// spotless:off
/**
 * Benchmark                                    (size)  Mode  Cnt    Score    Error  Units
 * NDArrayImplementationBenchmark.add               64  avgt    3    0.013 ±  0.001  ms/op
 * NDArrayImplementationBenchmark.add             1024  avgt    3    3.287 ±  0.078  ms/op
 * NDArrayImplementationBenchmark.add             4096  avgt    3   52.734 ±  0.736  ms/op
 * NDArrayImplementationBenchmark.addBroadcast      64  avgt    3    0.061 ±  0.001  ms/op
 * NDArrayImplementationBenchmark.addBroadcast    1024  avgt    3   15.271 ±  0.232  ms/op
 * NDArrayImplementationBenchmark.addBroadcast    4096  avgt    3  246.703 ± 10.302  ms/op
 * NDArrayImplementationBenchmark.addIterative      64  avgt    3    0.001 ±  0.001  ms/op
 * NDArrayImplementationBenchmark.addIterative    1024  avgt    3    0.182 ±  0.024  ms/op
 * NDArrayImplementationBenchmark.addIterative    4096  avgt    3    3.915 ±  0.016  ms/op
 * NDArrayImplementationBenchmark.addVector         64  avgt    3    0.001 ±  0.001  ms/op
 * NDArrayImplementationBenchmark.addVector       1024  avgt    3    0.182 ±  0.006  ms/op
 * NDArrayImplementationBenchmark.addVector       4096  avgt    3    3.905 ±  0.109  ms/op
 */
// spotless:on
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgsPrepend = ["--add-modules=jdk.incubator.vector", "-XX:-TieredCompilation"])
open class NDArrayImplementationBenchmark {
    @Param("64", "1024", "4096") private var size = 0

    private lateinit var A: FloatArray
    private lateinit var B: FloatArray

    private lateinit var NA: NDArray
    private lateinit var NB: NDArray

    @Setup
    fun setup() {
        A = FloatArray(size * size) { Random.nextFloat() }
        B = FloatArray(size * size) { Random.nextFloat() }

        NA = NDArrays.of(intArrayOf(size, size), A)
        NB = NDArrays.of(intArrayOf(size, size), B)
    }

    @Benchmark
    fun add() {
        NA.add(NB)
    }

    @Benchmark
    fun addVector() {
        NA.addVector(NB)
    }

    @Benchmark
    fun addIterative() {
        NA.addIterative(NB)
    }

    @Benchmark
    fun addBroadcast() {
        NA.addBroadcast(NB)
    }
}
