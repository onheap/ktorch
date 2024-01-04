package benchmarks

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import java.util.concurrent.TimeUnit
import jdk.incubator.vector.*
import kotlin.random.Random
import ndarray.NDArrays
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.openjdk.jmh.annotations.*

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgsPrepend = ["--add-modules=jdk.incubator.vector", "-XX:-TieredCompilation"])
open class BenchmarkLibraries {
    @Param("1024") private var size = 0

    private lateinit var A: FloatArray
    private lateinit var B: FloatArray

    private lateinit var manager: NDManager

    @Setup
    fun setup() {
        A = FloatArray(size * size) { Random.nextFloat() }
        B = FloatArray(size * size) { Random.nextFloat() }
    }

    @Setup(Level.Invocation)
    fun beforeEach() {
        manager = NDManager.newBaseManager()
    }

    @TearDown(Level.Invocation)
    fun afterEach() {
        // close manager for each benchmark,
        // otherwise it will produce an out of memory error
        manager.close()
    }

    @Benchmark
    fun MK_Matmul() {
        val MA = mk.ndarray(A, size, size)
        val MB = mk.ndarray(B, size, size)

        MA dot MB
    }

    @Benchmark
    fun NDArray_Matmul() {
        val NA = NDArrays.of(intArrayOf(size, size), A)
        val NB = NDArrays.of(intArrayOf(size, size), B)

        NA.matmul(NB)
    }

    @Benchmark
    fun DJL_Matmul() {
        val DA = manager.create(A, Shape(size.toLong(), size.toLong()))
        val DB = manager.create(B, Shape(size.toLong(), size.toLong()))

        DA.matMul(DB)
    }
}
