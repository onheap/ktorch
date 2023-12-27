package ndarray;

import jdk.incubator.vector.VectorOperators;

public enum ElementWiseReduceOperator {
    SUM(VectorOperators.ADD) {
        @Override
        float processSingle(float a, float b) {
            return a + b;
        }

        @Override
        float getInit(float[] A, int offset, int len) {
            return 0;
        }
    },

    MAX(VectorOperators.MAX) {
        @Override
        float processSingle(float a, float b) {
            return Math.max(a, b);
        }

        @Override
        float getInit(float[] A, int offset, int len) {
            return A[offset];
        }
    };

    public final VectorOperators.Associative vectorOperator;

    abstract float processSingle(float a, float b);

    abstract float getInit(float[] A, int offset, int len);

    ElementWiseReduceOperator(VectorOperators.Associative op) {
        this.vectorOperator = op;
    }
}
