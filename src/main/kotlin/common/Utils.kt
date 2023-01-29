package common

import core.value.Value


typealias Matrix<T> = List<List<T>>

class Utils {

//    data class Mnist(
//        val xTrain: Matrix<Double>,
//        val yTrain: Matrix<Double>,
//        val xTest: Matrix<Double>,
//        val yTest: Matrix<Double>,
//    )

    val sumValues: (Value, Value) -> Value = { acc, value -> acc + value }
}

