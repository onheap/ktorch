package core.tensor

import kotlin.collections.*
import ndarray.NDArrays
import ndarray.operator.FloatBinaryOperator

object Tensors {
    fun create(
        data: FloatArray,
        shape: IntArray = intArrayOf(data.size),
        requiresGrad: Boolean = false
    ): Tensor {
        return JvmTensor(NDArrays.of(shape, data), requiresGrad = requiresGrad)
    }

    fun create(data: Array<FloatArray>, requiresGrad: Boolean = false): Tensor {
        return JvmTensor(data = NDArrays.of(data), requiresGrad = requiresGrad)
    }

    fun onesLike(tensor: Tensor): Tensor {
        return JvmTensor(NDArrays.onesLike((tensor as JvmTensor).data))
    }

    fun zerosLike(tensor: Tensor): Tensor {
        return JvmTensor(NDArrays.zerosLike((tensor as JvmTensor).data))
    }

    fun zeros(vararg shape: Int): Tensor {
        return JvmTensor(NDArrays.of(shape))
    }

    fun createScalar(data: Float, requiresGrad: Boolean = false): Tensor {
        return JvmTensor(NDArrays.ofScalar(data), requiresGrad = requiresGrad)
    }

    fun stack(vararg tensors: Tensor): Tensor {
        return JvmTensor(
            data = NDArrays.stack(*tensors.map { (it as JvmTensor).data }.toTypedArray()))
    }

    fun perform(a: Tensor, b: Tensor, op: FloatBinaryOperator): Tensor {
        return JvmTensor(data = NDArrays.perform((a as JvmTensor).data, (b as JvmTensor).data, op))
    }
}
