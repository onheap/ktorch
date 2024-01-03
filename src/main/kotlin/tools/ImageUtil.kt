package tools

object ImageUtil {
    fun printGrayScaleImage(image: Array<FloatArray>) {
        val px = "  " // use two spaces to present one pixel

        for (row in image.indices) {
            for (col in image[row].indices) {
                val pixel = image[row][col]
                // Map the pixel value to a grayscale value between 0 and 23
                val grayscaleValue = (pixel * 23).toInt()

                // Use the grayscale value to generate an ANSI escape code for the background color
                val colorCode = String.format("\u001b[48;5;%dm%s", 232 + grayscaleValue, px)
                print(colorCode)
            }
            println("\u001b[0m") // Reset the color at the end of each line
        }
    }

    fun printMnistImageSmall(image: Array<FloatArray>) {
        var row = 0
        while (row < image.size) {
            for (col in image[row].indices) {
                val pixelTop = image[row][col]
                val pixelBottom: Float = if (row < image.size - 1) image[row + 1][col] else 1F
                when {
                    pixelTop < 0.5 && pixelBottom < 0.5 -> {
                        // Both pixels are light, print as full block
                        print("█")
                    }
                    pixelTop < 0.5 && pixelBottom >= 0.5 -> {
                        // Top pixel is light, bottom pixel is dark, print as top half block
                        print("▀")
                    }
                    pixelTop >= 0.5 && pixelBottom < 0.5 -> {
                        // Top pixel is dark, bottom pixel is light, print as bottom half block
                        print("▄")
                    }
                    else -> {
                        // Both pixels are dark, print as space
                        print(" ")
                    }
                }
            }
            println()
            row += 2
        }
    }
}
