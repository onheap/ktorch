package tools

import java.time.Duration
import java.time.Instant.now
import kotlin.random.Random
import kotlin.time.toKotlinDuration

object ProgressBar {
    private const val INCOMPLETE = " "
    private const val COMPLETE = "█"
    private const val BLOCKS = "▎▌▊"
    private const val RESET = "\r"

    fun loopFor(total: Int, taskName: String = "", task: (Int) -> Any?) {

        val start = now()

        var extMsg = ""
        printBar(taskName, 0, total, 0, extMsg)

        for (i in 0 until total) {
            val res = task(i)

            extMsg = if (res == Unit) "" else res.toString()

            val p = i * 100
            if (p % total == 0) {
                printBar(taskName, i, total, p / total, extMsg, Duration.between(start, now()))
            }
        }

        printBar(taskName, total, total, 100, extMsg, Duration.between(start, now()))
        println()
    }

    private fun printBar(
        taskName: String,
        curt: Int,
        total: Int,
        percent: Int,
        extMsg: String,
        duration: Duration = Duration.ZERO
    ) {
        val len = total.toString().length
        val countStr = String.format("%${len}d/%d", curt, total)
        val percentStr = String.format("%3d", percent)
        val durationStr = if (duration.isZero) "" else duration.toKotlinDuration().toString()

        when (percent % 4) {
            0 ->
                print(
                    "$RESET$taskName $countStr $percentStr%: │${COMPLETE.repeat(percent / 4)}${INCOMPLETE.repeat(25 - percent / 4)}│ $durationStr $extMsg")
            1 ->
                print(
                    "$RESET$taskName $countStr $percentStr%: │${COMPLETE.repeat(percent / 4)}${BLOCKS[0]}${INCOMPLETE.repeat(24 - percent / 4)}│ $durationStr $extMsg")
            2 ->
                print(
                    "$RESET$taskName $countStr $percentStr%: │${COMPLETE.repeat(percent / 4)}${BLOCKS[1]}${INCOMPLETE.repeat(24 - percent / 4)}│ $durationStr $extMsg")
            3 ->
                print(
                    "$RESET$taskName $countStr $percentStr%: │${COMPLETE.repeat(percent / 4)}${BLOCKS[2]}${INCOMPLETE.repeat(24 - percent / 4)}│ $durationStr $extMsg")
        }
    }
}

fun main(args: Array<String>) {
    ProgressBar.loopFor(300, "Random") {
        Thread.sleep(10)
        Random.nextFloat()
    }
}
