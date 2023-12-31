package core.tensor

import ndarray.NDArray

abstract class Operator {
    // parents
    val params = mutableListOf<Tensor>()
    private val savedTensor = mutableListOf<Tensor>()

    abstract fun forward(vararg tensors: Tensor): Tensor

    abstract fun backward(outputGrad: Tensor): List<Tensor>

    fun saveForBackward(vararg t: Tensor) {
        savedTensor.addAll(t)
    }

    fun savedTensor() = savedTensor

    operator fun invoke(vararg tensors: Tensor): Tensor {
        params.addAll(tensors)
        val ret = this.forward(*tensors)
        ret.operator = this
        return ret
    }

    companion object {
        val EMPTY =
            object : Operator() {
                override fun forward(vararg tensors: Tensor) =
                    throw IllegalStateException("Should not enter here")

                override fun backward(outputGrad: Tensor) =
                    throw IllegalStateException("Should not enter here")
            }
    }
}

abstract class JvmOperator : Operator() {
    fun saveForBackward(vararg a: NDArray) =
        super.saveForBackward(*a.map { JvmTensor(it) }.toTypedArray())

    fun savedNDArray() = savedTensor().map { (it as JvmTensor).data }
}

abstract class JvmBinaryOperator : JvmOperator() {
    abstract fun forward(left: NDArray, right: NDArray): NDArray

    abstract fun backward(
        outputGrad: NDArray,
        left: NDArray,
        right: NDArray
    ): Pair<NDArray, NDArray>

    override fun forward(vararg tensors: Tensor): Tensor {
        val res = forward((tensors[0] as JvmTensor).data, (tensors[1] as JvmTensor).data)
        return JvmTensor(res, params.any { it.requiresGrad })
    }

    override fun backward(outputGrad: Tensor): List<Tensor> {
        val (leftGrad, rightGrad) =
            backward(
                (outputGrad as JvmTensor).data,
                (params[0] as JvmTensor).data,
                (params[1] as JvmTensor).data)

        return listOf(JvmTensor(leftGrad), JvmTensor(rightGrad))
    }
}

abstract class JvmUnaryOperator : JvmOperator() {
    abstract fun forward(input: NDArray): NDArray

    abstract fun backward(outputGrad: NDArray, input: NDArray): NDArray

    override fun forward(vararg tensors: Tensor) =
        JvmTensor(forward((tensors[0] as JvmTensor).data), params.any { it.requiresGrad })

    override fun backward(outputGrad: Tensor): List<Tensor> {
        return listOf(
            JvmTensor(backward((outputGrad as JvmTensor).data, (params[0] as JvmTensor).data)))
    }
}
