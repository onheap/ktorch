package core.value

import kotlin.random.Random

// https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py
abstract class Module {
    abstract fun parameters(): List<Value>

    fun zeroGrad() = parameters().forEach { it.grad = 0.0 }
}

class Neuron(w: Int, val nonlin: Boolean = true) : Module() {
    private val weights: List<Value> = List(w) { Value(Random.nextDouble(-1.0, 1.0)) }
    private val bias: Value = Value(0.0)

    override fun parameters() = weights + bias

    operator fun invoke(input: List<Value>): Value {
        val act = (weights zip input).map { (xi, yi) -> xi * yi }.sum() + bias
        return if (this.nonlin) act.relu() else act
    }

    override fun toString(): String {
        return "${if (nonlin) "ReLU" else "Linear"}Neuron(${weights.size})"
    }
}

class Layer(nin: Int, nout: Int, nonlin: Boolean = true) : Module() {
    private val neurons: List<Neuron> = List(nout) { Neuron(nin, nonlin) }

    override fun parameters() = neurons.flatMap { it.parameters() }

    operator fun invoke(input: List<Value>): List<Value> {
        return neurons.map { neuron -> neuron(input) }
    }

    override fun toString(): String {
        return "\n  Layer of ${neurons.size} neurons: ${neurons}"
    }
}

class MLP(nin: Int, nouts: List<Int>) : Module() {
    private val layers: List<Layer>

    init {
        val sz = listOf(nin) + nouts
        this.layers = List(nouts.size) { i -> Layer(sz[i], sz[i + 1], i != nouts.size - 1) }
    }

    override fun parameters() = layers.flatMap { it.parameters() }

    operator fun invoke(initInput: List<Value>): List<Value> {
        return layers.fold(initInput) { input, layer -> layer(input) }
    }

    override fun toString(): String {
        return "MLP of ${layers.size} layers: $layers"
    }
}
