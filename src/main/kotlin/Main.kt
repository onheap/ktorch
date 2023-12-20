import core.benchmark.Benchmark

fun main(args: Array<String>) {
    val benchmark = Benchmark()
    benchmark.calFlopsMultik()
//    benchmark.calFlopsNDArraySimple()
//    benchmark.calFlopsNDArrayVector()
    benchmark.calFlopsNDArrayVectorConcurrent()
}