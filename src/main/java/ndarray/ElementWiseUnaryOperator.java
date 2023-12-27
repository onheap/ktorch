package ndarray;

import jdk.incubator.vector.VectorOperators;

public enum ElementWiseUnaryOperator {
    LOG(VectorOperators.LOG) {
        @Override
        float processSingle(float f) {
            return (float) Math.log(f);
        }
    },
    EXP(VectorOperators.EXP) {
        @Override
        float processSingle(float f) {
            return (float) Math.exp(f);
        }
    },
    ;

    public final VectorOperators.Unary vectorOperator;

    abstract float processSingle(float f);

    ElementWiseUnaryOperator(VectorOperators.Unary op) {
        this.vectorOperator = op;
    }
}
