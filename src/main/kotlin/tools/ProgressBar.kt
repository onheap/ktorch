package tools

import java.time.Duration
import java.time.Instant.now
import kotlin.random.Random
import kotlin.time.toKotlinDuration

object ProgressBar {
    private const val UNCOMPLETED = " "
    private const val COMPLETED = "█"
    private const val BLOCKS = " ▏▎▍▌▋▊▉"

    private const val RESET = "\r"

    fun loopFor(total: Int, taskName: String = "", task: (Int) -> Any?) {

        val start = now()

        var extMsg = ""
        printBar(taskName, 0, total, extMsg, Duration.ZERO)

        for (i in 0 until total) {
            val res = task(i)

            extMsg = if (res == Unit) "" else res.toString()

            printBar(taskName, i, total, extMsg, Duration.between(start, now()))
        }

        printBar(taskName, total, total, extMsg, Duration.between(start, now()))
        println()
    }

    private fun printBar(
        taskName: String,
        curt: Int,
        total: Int,
        extMsg: String,
        duration: Duration
    ) {

        val step = (curt * 200) / total // the progress bar is divided to 200 steps
        val len = total.toString().length
        val countStr = String.format("%${len}d/%d", curt, total)
        val percentStr = String.format("%3d", (curt * 100) / total)
        val durationStr = if (duration.isZero) "" else duration.toKotlinDuration().toString()

        val c = step / BLOCKS.length
        when (val i = step % BLOCKS.length) {
            0 ->
                print(
                    "$RESET$taskName $countStr $percentStr%: │${COMPLETED.repeat(c)}${UNCOMPLETED.repeat(25 - c)}│ $durationStr $extMsg")
            else -> {
                print(
                    "$RESET$taskName $countStr $percentStr%: │${COMPLETED.repeat(c)}${BLOCKS[i]}${UNCOMPLETED.repeat(24 - c)}│ $durationStr $extMsg")
            }
        }
    }
}

fun main(args: Array<String>) {
    ProgressBar.loopFor(323, "Random") {
        Thread.sleep(80)
        Random.nextFloat()
    }
}
