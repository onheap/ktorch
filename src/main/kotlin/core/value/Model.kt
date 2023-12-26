package core.value

interface Model {
    fun prediction(initInput: List<Value>): Value

    fun parameters(): List<Value>

    fun zeroGrad() = parameters().forEach { it.grad = 0.0 }
}

class RawModel(nIn: Int, nOuts: List<Int>) : Model {

    // Neuron:  List<Value>  weights + bias
    // Layer:   List<Neuron>
    // Network: List<Layer>

    private val layers: List<List<List<Value>>>

    init {
        val sz = listOf(nIn) + nOuts
        val layersShapes = sz.windowed(2) { it.first() to it.last() }
        val layersIndexes =
            layersShapes.fold(listOf(0)) { idxes, (nin, nout) ->
                idxes + (idxes.last() + nin * nout)
            }

        val allParams = List(layersIndexes.last()) { i -> Value(rand(i)) }

        this.layers =
            layersShapes.mapIndexed { idx, (nin, nout) ->
                val startIdx = layersIndexes[idx]
                val endIdx = layersIndexes[idx + 1]
                allParams.slice(startIdx until endIdx).chunked(nin) { weights ->
                    weights + Value(0.0)
                }
            }
    }

    override fun prediction(initInput: List<Value>): Value {
        return layers
            .foldIndexed(initInput) { layerIdx, input, layer ->
                layer.map { neuron ->
                    // weights * input + bias
                    val act =
                        (input zip neuron.take(input.size)).map { (xi, yi) -> xi * yi }.sum() +
                            neuron.last()
                    if (layerIdx == layers.size - 1) act else act.relu()
                }
            }
            .single()
    }

    override fun parameters(): List<Value> {
        return layers.flatMap { it.flatten() }
    }

    override fun toString(): String {
        return layers.toString()
    }

    fun rand(i: Int): Double {
        val l = (1103515245L * (i + 1) + 12345) % (4294967296)
        val rd = l / 4294967295.0
        return (1 - -1) * rd + -1
    }
}
