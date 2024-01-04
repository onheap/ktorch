import example.mnist.MNIST

// ./gradlew run --args=mnist
fun main(args: Array<String>) {
    if (args.isEmpty()) {
        System.err.println("no args")
        return
    }

    when (val task = args[0].lowercase()) {
        "mnist" -> {
            MNIST.train()
            MNIST.eval()
        }
        else -> {
            System.err.println("unknown task $task")
        }
    }
}
