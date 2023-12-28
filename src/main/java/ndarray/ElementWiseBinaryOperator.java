package ndarray;

import jdk.incubator.vector.VectorOperators;

public enum ElementWiseBinaryOperator {
    ADD(VectorOperators.ADD) {
        @Override
        float processSingle(float a, float b) {
            return a + b;
        }
    },

    SUB(VectorOperators.SUB) {
        @Override
        float processSingle(float a, float b) {
            return a - b;
        }
    },

    MUL(VectorOperators.MUL) {
        @Override
        float processSingle(float a, float b) {
            return a * b;
        }
    },

    DIV(VectorOperators.DIV) {
        @Override
        float processSingle(float a, float b) {
            return a / b;
        }
    },

    MAX(VectorOperators.MAX) {
        @Override
        float processSingle(float a, float b) {
            return Math.max(a, b);
        }
    },

    MIN(VectorOperators.MIN) {
        @Override
        float processSingle(float a, float b) {
            return Math.min(a, b);
        }
    };

    public final VectorOperators.Binary vectorOperator;

    abstract float processSingle(float a, float b);

    ElementWiseBinaryOperator(VectorOperators.Binary op) {
        this.vectorOperator = op;
    }
}
