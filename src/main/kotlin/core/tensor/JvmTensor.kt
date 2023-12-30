package core.tensor

import ndarray.NDArray

class JvmTensor(
    val data: NDArray,
    override val requiresGrad: Boolean = false,
    override var operator: Operator = Operator.EMPTY,
    override var grad: Tensor? = null,
) : Tensor {
    override fun shape(): IntArray = data.shape

    override fun size(): Int = data.size

    override fun get(vararg indices: Int): Float {
        return data[indices]
    }

    override fun plus(x: Tensor): Tensor {
        return object : JvmBinaryOperator() {
            override fun forward(left: NDArray, right: NDArray): NDArray {
                return left.add(right)
            }

            override fun backward(outputGrad: NDArray, left: NDArray, right: NDArray) =
                outputGrad to outputGrad
        }(this, x)
    }

    override fun minus(x: Tensor): Tensor = TODO()

    override fun times(x: Tensor): Tensor {
        return object : JvmBinaryOperator() {
            override fun forward(left: NDArray, right: NDArray) = left.mul(right)

            override fun backward(outputGrad: NDArray, left: NDArray, right: NDArray) =
                right.mul(outputGrad) to left.mul(outputGrad)
        }(this, x)
    }

    override fun div(x: Tensor): Tensor = TODO()

    override fun exp(): Tensor = TODO()

    override fun log(): Tensor = TODO()

    override fun relu(): Tensor {
        return object : JvmUnaryOperator() {
            override fun forward(input: NDArray) = input.maximum(0F)

            override fun backward(outputGrad: NDArray, input: NDArray) = outputGrad.minimum(0F)
        }(this)
    }

    override fun transpose(): Tensor = TODO()

    override fun sum(): Tensor {
        return object : JvmUnaryOperator() {
            override fun forward(input: NDArray) = input.sum()

            override fun backward(outputGrad: NDArray, input: NDArray) =
                outputGrad.mul(NDArray.onesLike(input))
        }(this)
    }

    override fun matmul(x: Tensor): Tensor {
        return object : JvmBinaryOperator() {
            override fun forward(left: NDArray, right: NDArray) = left.matmul(right)

            override fun backward(
                outputGrad: NDArray,
                left: NDArray,
                right: NDArray
            ): Pair<NDArray, NDArray> {
                return Pair(
                    first = outputGrad.matmul(right.transpose()),
                    second = outputGrad.transpose().matmul(left).transpose())
            }
        }(this, x)
    }

    override fun logSoftmax(): Tensor {
        return object : JvmUnaryOperator() {
            override fun forward(input: NDArray): NDArray {
                // logsumexp
                val x = input
                val c = x.max(1)
                val logsumexp = x.sub(c.reshape(-1, 1)).exp().sum(1).log().add(c)

                val output = x.sub(logsumexp.reshape(-1, 1))
                saveForBackward(output)
                return output
            }

            override fun backward(outputGrad: NDArray, input: NDArray): NDArray {
                val (output) = this.savedNDArray()
                return outputGrad.sub(output.exp().mul(outputGrad.sum(1).reshape(-1, 1)))
            }
        }(this)
    }

    override fun toString(): String {
        return data.toString()
    }
}
