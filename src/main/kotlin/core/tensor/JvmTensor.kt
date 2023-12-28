package core.tensor

import ndarray.NDArray

typealias NDTensor = Tensor<NDArray>

typealias NDBiOperator = BinaryOperator<NDArray>

typealias NDUiOperator = UnaryOperator<NDArray>

class JvmTensor(
    override var data: NDArray,
    override var operator: Operator<NDArray>,
    override var grad: NDArray = NDArray.EMPTY,
) : NDTensor {
    override fun shape(): IntArray = data.shape

    override fun size(): Int = data.size

    override fun plus(x: NDTensor): NDTensor {
        return object : NDBiOperator() {
            override fun forward(left: NDTensor, right: NDTensor): NDTensor {
                return JvmTensor(left.data.add(right.data), this)
            }

            override fun backward(outputGrad: NDArray, left: NDTensor, right: NDTensor) {
                left.grad = outputGrad
                right.grad = outputGrad
            }
        }(this, x)
    }

    override fun minus(x: NDTensor): NDTensor = TODO()

    override fun times(x: NDTensor): NDTensor {
        return object : NDBiOperator() {
            override fun forward(left: NDTensor, right: NDTensor): NDTensor {
                return JvmTensor(left.data.mul(right.data), this)
            }

            override fun backward(outputGrad: NDArray, left: NDTensor, right: NDTensor) {
                left.grad = left.data.mul(outputGrad)
                right.grad = right.data.mul(outputGrad)
            }
        }(this, x)
    }

    override fun div(x: NDTensor): NDTensor = TODO()

    override fun exp(): NDTensor = TODO()

    override fun log(): NDTensor = TODO()

    override fun relu(): NDTensor {
        return object : NDUiOperator() {
            override fun forward(input: NDTensor): NDTensor {
                return JvmTensor(input.data.maximum(0F), this)
            }

            override fun backward(outputGrad: NDArray, input: NDTensor) {
                input.grad = outputGrad.maximum(0F)
            }
        }(this)
    }

    override fun transpose(): NDTensor = TODO()

    override fun sum(): NDTensor {
        return object : NDUiOperator() {
            override fun forward(input: NDTensor): NDTensor {
                return JvmTensor(input.data.sum(), this)
            }

            override fun backward(outputGrad: NDArray, input: NDTensor) {
                input.grad = outputGrad.mul(NDArray.onesLike(input.data))
            }
        }(this)
    }

    override fun matmul(x: NDTensor): NDTensor {
        return object : NDBiOperator() {
            override fun forward(left: NDTensor, right: NDTensor): NDTensor {
                return JvmTensor(left.data.matmul(right.data), this)
            }

            override fun backward(outputGrad: NDArray, left: NDTensor, right: NDTensor) {
                left.grad = outputGrad.matmul(right.data.transpose())
                right.grad = outputGrad.transpose().matmul(left.data).transpose()
            }
        }(this, x)
    }

    override fun logSoftmax(): NDTensor {
        return object : NDUiOperator() {
            override fun forward(input: NDTensor): NDTensor {
                // logsumexp
                val x = input.data
                val c = x.max(1)
                val logsumexp = x.sub(c.reshape(intArrayOf(-1, 1))).exp().sum(1).log().add(c)

                val output = x.sub(logsumexp.reshape(intArrayOf(-1, 1)))
                saveForBackward(output)
                return JvmTensor(output, this)
            }

            override fun backward(outputGrad: NDArray, input: NDTensor) {
                val (output) = this.savedData
                input.grad =
                    outputGrad.sub(output.exp().mul(outputGrad.sum(1).reshape(intArrayOf(-1, 1))))
            }
        }(this)
    }
}
