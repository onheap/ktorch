package core.tensor

import ndarray.NDArray
import ndarray.NDArrays

class JvmTensor(
    var data: NDArray,
    override var requiresGrad: Boolean = false,
    override var operator: Operator = Operator.EMPTY,
    override var grad: Tensor? = null,
) : Tensor {
    override fun shape(): IntArray = data.shape

    override fun size(): Int = data.size

    override fun toArray(): FloatArray = data.toArray()

    override fun toMatrix(): Array<FloatArray> = data.toMatrix()

    override fun isScalar(): Boolean = data.isScalar

    override fun asScalar(): Float = data.asScalar()

    override fun get(vararg indices: Int): Float = data[indices]

    override fun set(vararg indices: Int, v: Float) = data.set(indices, v)

    override fun reshape(vararg newShape: Int): Tensor = JvmTensor(data.reshape(*newShape))

    override fun plus(x: Tensor): Tensor {
        return object : JvmBinaryOperator() {
            override fun forward(left: NDArray, right: NDArray): NDArray {
                return left.add(right)
            }

            override fun backward(
                outputGrad: NDArray,
                left: NDArray,
                right: NDArray
            ): Pair<NDArray, NDArray> {
                return Pair(
                    sumBroadcastDimsGrad(outputGrad, left), sumBroadcastDimsGrad(outputGrad, right))
            }
        }(this, x)
    }

    override fun minus(x: Tensor): Tensor {
        return object : JvmBinaryOperator() {
            override fun forward(left: NDArray, right: NDArray): NDArray {
                return left.sub(right)
            }

            override fun backward(
                outputGrad: NDArray,
                left: NDArray,
                right: NDArray
            ): Pair<NDArray, NDArray> {
                return Pair(
                    sumBroadcastDimsGrad(outputGrad, left),
                    sumBroadcastDimsGrad(outputGrad.mul(NDArrays.ofScalar(-1F)), right))
            }
        }(this, x)
    }

    override fun times(x: Tensor): Tensor {
        return object : JvmBinaryOperator() {
            override fun forward(left: NDArray, right: NDArray) = left.mul(right)

            override fun backward(
                outputGrad: NDArray,
                left: NDArray,
                right: NDArray
            ): Pair<NDArray, NDArray> {
                return Pair(
                    sumBroadcastDimsGrad(right.mul(outputGrad), left),
                    sumBroadcastDimsGrad(left.mul(outputGrad), right))
            }
        }(this, x)
    }

    override fun div(x: Tensor): Tensor = TODO()

    override fun exp(): Tensor = TODO()

    override fun log(): Tensor = TODO()

    override fun relu(): Tensor {
        return object : JvmUnaryOperator() {
            override fun forward(input: NDArray) = input.maximum(0F)

            override fun backward(outputGrad: NDArray, input: NDArray): NDArray {
                return NDArrays.perform(outputGrad, input) { a, b -> if (b < 0) 0F else a }
            }
        }(this)
    }

    override fun mean(): Tensor {
        return sum().mul(Tensors.createScalar(1F / size(), requiresGrad))
    }

    override fun transpose(): Tensor = TODO()

    override fun sum(): Tensor {
        return object : JvmUnaryOperator() {
            override fun forward(input: NDArray) = input.sum()

            override fun backward(outputGrad: NDArray, input: NDArray) =
                outputGrad.mul(NDArrays.onesLike(input))
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
                val (output) = this.savedNDArrays()
                return outputGrad.sub(output.exp().mul(outputGrad.sum(1).reshape(-1, 1)))
            }
        }(this)
    }

    override fun argmax(): Tensor = JvmTensor(data.argmax())

    override fun argmax(dim: Int, keepDims: Boolean): Tensor = JvmTensor(data.argmax(dim, keepDims))

    private fun sumBroadcastDimsGrad(grad: NDArray, param: NDArray): NDArray {
        val paramShape = param.shape
        val gradShape = grad.shape

        var finalGrad = grad
        for (k in gradShape.indices) {
            val i = gradShape.size - k - 1
            val j = paramShape.size - k - 1
            if (j >= 0) {
                if (gradShape[i] != paramShape[j]) {
                    finalGrad = finalGrad.sum(i, true)
                }
            } else {
                finalGrad = finalGrad.sum(0, false)
            }
        }
        return finalGrad
    }

    override fun toString(): String {
        return data.toString()
    }

    private fun copyToThis(x: Tensor) {
        val ret = x as JvmTensor
        this.data = ret.data
        this.requiresGrad = ret.requiresGrad
        this.operator = ret.operator
        this.grad = ret.grad
    }
}
