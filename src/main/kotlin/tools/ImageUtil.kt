package tools

// https://github.com/Nellousan/px2ansi/blob/main/px2ansi.py
class ImageUtil {}

const val ANSI_RESET = "\u001B[0m"
const val ANSI_BLACK = "\u001B[30m"
const val ANSI_RED = "\u001B[31m"
const val ANSI_GREEN = "\u001B[32m"

fun main(args: Array<String>) {

    println(ANSI_RED + "red" + ANSI_RESET + " normal " + ANSI_GREEN + "green" + ANSI_RESET)
}
