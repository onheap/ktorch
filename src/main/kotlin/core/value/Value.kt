package core.value

import kotlin.math.pow

// inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
data class Value(
    var data: Double,
    var grad: Double = 0.0,
    var _prev: Set<Value> = setOf(),
    var _backward: (Value) -> Unit = {},
) {

    override fun toString(): String {
        return "Value(data=$data, grad=$grad)"
    }
    operator fun plus(other: Value): Value {
        return Value(data + other.data, _prev = setOf(this, other), _backward = {
            this.grad += it.grad
            other.grad += it.grad
        })
    }


    operator fun times(other: Value): Value {
        return Value(data * other.data, _prev = setOf(this, other), _backward = {
            this.grad += other.data * it.grad
            other.grad += this.data * it.grad
        })
    }

    // never infix `pow` operation, because it has precedence issues
    // infix fun `**`(other: Double) = this.pow(other)
    fun pow(other: Double): Value {
        return Value(this.data.pow(other), _prev = setOf(this), _backward = {
            this.grad += (other * this.data.pow(other - 1)) * it.grad
        })
    }

    fun relu(): Value {
        return Value(if (data > 0) data else 0.0, _prev = setOf(this), _backward = {
            this.grad += it.grad * if (it.data > 0) 1.0 else 0.0
        })
    }


    operator fun unaryMinus() = this * -1.0
    operator fun minus(other: Value) = this + -other
    operator fun div(other: Value) = this * other.pow(-1.0)


    fun backward() {
        val topo = mutableListOf<Value>()
        val visited = mutableSetOf<Value>()


        fun buildTopo(v: Value) {
            if (v in visited) {
                return
            }

            visited.add(v)

            for (c in v._prev) {
                buildTopo(c)
            }
            topo.add(v)
        }

        buildTopo(this)

        this.grad = 1.0
        for (v in topo.reversed()) {
            v._backward(v)
        }
    }
}

operator fun Double.plus(other: Value) = Value(this).plus(other)
operator fun Double.minus(other: Value) = Value(this).minus(other)
operator fun Double.times(other: Value) = Value(this).times(other)
operator fun Double.div(other: Value) = Value(this).div(other)

operator fun Value.plus(other: Double) = this.plus(Value(other))
operator fun Value.minus(other: Double) = this.minus(Value(other))
operator fun Value.times(other: Double) = this.times(Value(other))
operator fun Value.div(other: Double) = this.div(Value(other))

operator fun Int.plus(other: Value) = Value(this.toDouble()).plus(other)
operator fun Int.minus(other: Value) = Value(this.toDouble()).minus(other)
operator fun Int.times(other: Value) = Value(this.toDouble()).times(other)
operator fun Int.div(other: Value) = Value(this.toDouble()).div(other)

operator fun Value.plus(other: Int) = this.plus(Value(other.toDouble()))
operator fun Value.minus(other: Int) = this.minus(Value(other.toDouble()))
operator fun Value.times(other: Int) = this.times(Value(other.toDouble()))
operator fun Value.div(other: Int) = this.div(Value(other.toDouble()))
fun Value.pow(other: Int) = this.pow(other.toDouble())

fun List<Value>.sum():Value {
    var res = Value(0.0)
    for (v in this) {
        res += v
    }
    return res

//    return this.reduce { acc, value -> acc + value }
}
