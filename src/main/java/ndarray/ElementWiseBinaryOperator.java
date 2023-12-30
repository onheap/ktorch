package ndarray;

import static ndarray.Util.*;

import jdk.incubator.vector.VectorOperators;

public enum ElementWiseBinaryOperator {
    ADD(VectorOperators.ADD, (a, b) -> a + b);

    public final VectorOperators.Binary vectorOperator;

    public final FloatBinaryOperator singleOperator;

    ElementWiseBinaryOperator(
            VectorOperators.Binary vectorOperator, FloatBinaryOperator singleOperator) {
        this.vectorOperator = vectorOperator;
        this.singleOperator = singleOperator;
    }
}
