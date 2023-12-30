package core.tensor

import java.lang.IllegalStateException
import ndarray.NDArray

interface Tensor {
    companion object {
        fun create(
            data: FloatArray,
            shape: IntArray = intArrayOf(data.size),
            requiresGrad: Boolean = false
        ): Tensor {
            return JvmTensor(NDArray.of(shape, data), requiresGrad = requiresGrad)
        }

        fun onesLike(tensor: Tensor): Tensor {
            return JvmTensor(NDArray.onesLike((tensor as JvmTensor).data))
        }

        fun zerosLike(tensor: Tensor): Tensor {
            return JvmTensor(NDArray.zerosLike((tensor as JvmTensor).data))
        }
    }

    var grad: Tensor?
    var operator: Operator
    val requiresGrad: Boolean

    fun shape(): IntArray

    fun size(): Int

    fun get(vararg indices: Int): Float

    operator fun plus(x: Tensor): Tensor

    operator fun minus(x: Tensor): Tensor

    operator fun times(x: Tensor): Tensor

    operator fun div(x: Tensor): Tensor

    fun matmul(x: Tensor): Tensor

    fun exp(): Tensor

    fun log(): Tensor

    fun relu(): Tensor

    fun transpose(): Tensor

    fun sum(): Tensor

    fun logSoftmax(): Tensor

    fun deepWalk(): List<Tensor> {
        val topo = mutableListOf<Tensor>()
        val visited = mutableSetOf<Tensor>()

        fun buildTopo(t: Tensor) {
            if (t in visited) {
                return
            }

            visited.add(t)

            for (p in operator.params) {
                buildTopo(p)
            }
            topo.add(t)
        }

        buildTopo(this)

        return topo
    }

    fun backward() {
        // backward can only be called for scalar tensors
        this.grad = onesLike(this)

        for (t in deepWalk().reversed()) {
            check(t.requiresGrad) { throw IllegalStateException("requireGrad is false") }
            t.grad ?: throw IllegalStateException("grad is null")

            if (t.operator.params.isEmpty()) {
                continue
            }

            t.operator.params.zip(t.operator.backward(t.grad!!)).forEach { (p, g) ->
                if (p.grad == null) {
                    p.grad = g
                } else {
                    p.grad = p.grad!! + g
                }
            }
        }
    }

    fun zeroGrad() {
        this.grad = zerosLike(this)
    }
}
