package core.tensor

interface Tensor {

    var grad: Tensor?
    var operator: Operator
    var requiresGrad: Boolean

    fun shape(): IntArray

    fun size(): Int

    fun toArray(): FloatArray

    fun toMatrix(): Array<FloatArray>

    fun isScalar(): Boolean

    fun asScalar(): Float

    operator fun get(vararg indices: Int): Float

    operator fun set(vararg indices: Int, v: Float)

    fun reshape(vararg newShape: Int): Tensor

    operator fun plus(x: Tensor): Tensor

    operator fun minus(x: Tensor): Tensor

    operator fun times(x: Tensor): Tensor

    operator fun div(x: Tensor): Tensor

    fun add(x: Tensor): Tensor = this.plus(x)

    fun sub(x: Tensor): Tensor = this.minus(x)

    fun mul(x: Tensor): Tensor = this.times(x)

    fun matmul(x: Tensor): Tensor

    fun exp(): Tensor

    fun log(): Tensor

    fun relu(): Tensor

    fun mean(): Tensor

    fun transpose(): Tensor

    fun sum(): Tensor

    fun logSoftmax(): Tensor

    fun argmax(): Tensor

    fun argmax(dim: Int, keepDims: Boolean = false): Tensor

    fun deepWalk(): List<Tensor> {
        val topo = mutableListOf<Tensor>()
        val visited = mutableSetOf<Tensor>()

        fun buildTopo(t: Tensor) {
            if (t in visited) {
                return
            }

            visited.add(t)

            for (p in t.operator.params) {
                buildTopo(p)
            }
            topo.add(t)
        }

        buildTopo(this)
        return topo
    }

    fun backward() {
        // backward can only be called for scalar tensors
        this.grad = Tensors.onesLike(this)

        for (t in deepWalk().reversed()) {
            if (!t.requiresGrad) continue

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
        if (this.grad != null) {
            this.grad = Tensors.zerosLike(this.grad!!)
        }
    }
}
