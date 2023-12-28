package ndarray;

import jdk.incubator.vector.VectorOperators;

public enum ElementWiseReduceOperator {
    SUM(VectorOperators.ADD) {
        @Override
        float processSingle(float a, float b) {
            return a + b;
        }

        @Override
        float getInit() {
            return 0;
        }
    },

    MAX(VectorOperators.MAX) {
        @Override
        float processSingle(float a, float b) {
            return Math.max(a, b);
        }

        @Override
        float getInit() {
            return Float.NEGATIVE_INFINITY;
        }
    };

    public final VectorOperators.Associative vectorOperator;

    abstract float processSingle(float a, float b);

    abstract float getInit();

    ElementWiseReduceOperator(VectorOperators.Associative op) {
        this.vectorOperator = op;
    }
}
