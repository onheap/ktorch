package common

class RandGen {
    private var cnt = 0

    fun getNext(lowest: Int, highest: Int): Double {
        val r = rand(cnt, lowest, highest)
        cnt++
        return r
    }

    // https://en.wikipedia.org/wiki/Linear_congruential_generator
    fun rand(i: Int, lowest: Int, highest: Int): Double {
        val l = (1103515245L * (i + 1) + 12345) % (4294967296)
        val rd = l / 4294967295.0
        return (highest - lowest) * rd + lowest
    }
}
