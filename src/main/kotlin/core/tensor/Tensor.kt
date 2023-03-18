package core.tensor

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.times

// inspired by https://github.com/geohot/tinygrad/blob/master/tinygrad/tensor.py
// https://github.com/geohot/tinygrad/blob/91a352a8e2697828a4b1eafa2bdc1a9a3b7deffa/tinygrad/tensor.py

typealias Matrix = D2Array<Float>

class Tensor(
    var data: Matrix,
    var grad: Matrix?,
) {

}