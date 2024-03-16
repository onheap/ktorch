package ndarray.operator;

import static ndarray.utils.Util.SPECIES;
import static ndarray.utils.Util.SPECIES_LEN;
import static ndarray.utils.Util.elementwiseOperable;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import ndarray.NDArray;
import ndarray.NDArrays;

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

    public NDArray performElementwise(NDArray ndArray) {
        assert elementwiseOperable(ndArray);
        NDArray res = NDArrays.zerosLike(ndArray);
        float[] A = ndArray.getData();
        float[] B = res.getData();

        int len = A.length;
        int i = 0;
        for (; i < SPECIES.loopBound(len); i += SPECIES_LEN) {
            var va = FloatVector.fromArray(SPECIES, A, i);
            va.lanewise(vectorOperator).intoArray(B, i);
        }

        for (; i < len; i++) {
            B[i] = processSingle(A[i]);
        }

        return res;
    }
}
