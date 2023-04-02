interface Tensor {
    fun getShape(): IntArray

    fun getStrides(): IntArray

    fun numel(): Int

    fun view(vararg shape: Int): Tensor

    fun flatten(dimension: Int): Tensor

    fun exp(): Tensor

    fun add(x: Tensor): Tensor
}