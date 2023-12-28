package benchmarks

import java.util.concurrent.TimeUnit
import jdk.incubator.vector.*
import kotlin.random.Random
import ndarray.NDArray
import org.openjdk.jmh.annotations.*

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgsPrepend = ["--add-modules=jdk.incubator.vector", "-XX:-TieredCompilation"])
open class NDArrayImplementationBenchmark {
    @Param("1024") private var size = 0

    private lateinit var A: FloatArray
    private lateinit var B: FloatArray

    private lateinit var NA: NDArray
    private lateinit var NB: NDArray

    @Setup
    fun setup() {
        A = FloatArray(size * size) { Random.nextFloat() }
        B = FloatArray(size * size) { Random.nextFloat() }

        NA = NDArray.of(intArrayOf(size, size), A)
        NB = NDArray.of(intArrayOf(size, size), B)
    }

    @Benchmark
    fun addOld() {
        NA.add(NB)
    }

    @Benchmark
    fun addNew() {
        NA.addNew(NB)
    }
    //
    //    @Benchmark
    //    fun subOld() {
    //        NA.sub(NB)
    //    }
    //
    //    @Benchmark
    //    fun subNew() {
    //        NA.subNew(NB)
    //    }
    //
    //    @Benchmark
    //    fun MDArray_logOld() {
    //        NA.log()
    //    }
    //
    //    @Benchmark
    //    fun MDArray_logNew() {
    //        NA.logNew()
    //    }
}
