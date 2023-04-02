package core.tensor

import Tensor


class JvmTensor(
    private val shape: IntArray,
    private val data: FloatArray,
) : Tensor {
    companion object {
        private fun getNumElements(shape: IntArray): Int = shape.reduce { acc, curt -> acc * curt }

        private fun makeStrides(shape: IntArray): IntArray {
            return shape.foldRightIndexed(IntArray(shape.size)) { idx: Int, curt: Int, strides: IntArray ->
                strides[idx] = if (idx == shape.size - 1) {
                    1
                } else {
                    curt * strides[idx + 1]
                }
                return strides
            }
        }
    }

    private val strides: IntArray = makeStrides(shape)

    constructor(shape: IntArray) : this(
        shape = shape, data = FloatArray(getNumElements(shape))
    )

    override fun getShape(): IntArray {
        return this.shape
    }

    override fun getStrides(): IntArray {
        return this.strides
    }

    override fun numel(): Int {
        return data.size
    }

    override fun view(vararg shape: Int): Tensor {
        TODO("Not yet implemented")
    }

    override fun flatten(dimension: Int): Tensor {
        TODO("Not yet implemented")
    }

    override fun exp(): Tensor {
        val copy = FloatArray(numel()) { i ->
            kotlin.math.exp(data[i])
        }
        return JvmTensor(shape, copy)
    }

    override fun add(x: Tensor): Tensor {
        val shape = broadcastShapes(x)
        val res = JvmTensor(shape)



    }


    private fun getFlatIndex(index: IntArray): Int {
        return index.reduceIndexed { idx, acc, curt -> curt * strides[idx] + acc }
    }


    private fun broadcastShapes(other: Tensor): IntArray {
        val (shapeA, shapeB) = if (this.shape.size >= other.getShape().size) {
            this.getShape() to other.getShape()
        } else {
            other.getShape() to this.getShape()
        }

        val broadcastShape = IntArray(shapeA.size)
        for (i in shapeA.indices) {
            val elementA = shapeA[shapeA.size - 1 - i]
            val elementB = if (i < shapeB.size) shapeB[shapeB.size - 1 - i] else -1
            val dimSize = if (elementA == elementB || elementB == 1 || elementB == -1) {
                elementA
            } else if (elementA == 1) {
                elementB
            } else {
                throw IllegalArgumentException(
                    "Shapes are not broadcast-able: shapeA: $shapeA, shapeB: $shapeB"
                )
            }
            broadcastShape[broadcastShape.size - 1 - i] = dimSize
        }
        return broadcastShape
    }


}