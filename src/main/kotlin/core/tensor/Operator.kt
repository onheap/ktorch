package core.tensor

abstract class Operator<T> {
    lateinit var params: Array<out Tensor<T>>
    val savedData = mutableListOf<T>()

    abstract fun forward(vararg tensors: Tensor<T>): Tensor<T>

    abstract fun backward(outputGrad: T)

    fun saveForBackward(vararg t: T) {
        savedData.addAll(t)
    }

    operator fun invoke(vararg tensors: Tensor<T>): Tensor<T> {
        params = tensors
        val ret = this.forward(*tensors)
        ret.operator = this
        return ret
    }
}

abstract class BinaryOperator<T> : Operator<T>() {
    abstract fun forward(left: Tensor<T>, right: Tensor<T>): Tensor<T>

    abstract fun backward(outputGrad: T, left: Tensor<T>, right: Tensor<T>)

    override fun forward(vararg tensors: Tensor<T>) = forward(tensors[0], tensors[1])

    override fun backward(outputGrad: T) = backward(outputGrad, params[0], params[1])
}

abstract class UnaryOperator<T> : Operator<T>() {
    abstract fun forward(input: Tensor<T>): Tensor<T>

    abstract fun backward(outputGrad: T, input: Tensor<T>)

    override fun forward(vararg tensors: Tensor<T>) = forward(tensors[0])

    override fun backward(outputGrad: T) = backward(outputGrad, params[0])
}
