package core.tensor

interface Tensor<T> {

    var data: T
    var grad: T
    var operator: Operator<T>

    fun shape(): IntArray

    fun size(): Int

    operator fun plus(x: Tensor<T>): Tensor<T>

    operator fun minus(x: Tensor<T>): Tensor<T>

    operator fun times(x: Tensor<T>): Tensor<T>

    operator fun div(x: Tensor<T>): Tensor<T>

    fun matmul(x: Tensor<T>): Tensor<T>

    fun exp(): Tensor<T>

    fun log(): Tensor<T>

    fun relu(): Tensor<T>

    fun transpose(): Tensor<T>

    fun sum(): Tensor<T>

    fun logSoftmax(): Tensor<T>
}
