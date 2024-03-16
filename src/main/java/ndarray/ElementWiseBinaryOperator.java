package ndarray;

import static ndarray.Util.*;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public enum ElementWiseBinaryOperator {
    ADD(VectorOperators.ADD) {
        @Override
        public float processSingle(float a, float b) {
            return a + b;
        }
    };

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int SPECIES_LEN = SPECIES.length();

    public final VectorOperators.Binary vectorOperator;

    public abstract float processSingle(float a, float b);

    ElementWiseBinaryOperator(
            VectorOperators.Binary vectorOperator) {
        this.vectorOperator = vectorOperator;
    }

    public NDArray performElementwise(NDArray a, NDArray b) {
        NDArray res = NDArrays.zerosLike(a);

        float[] A = a.data;
        float[] B = b.data;
        float[] C = res.data;

        int i = 0;
        for (; i < SPECIES.loopBound(C.length); i += SPECIES_LEN) {
            var va = FloatVector.fromArray(SPECIES, A, i);
            var vb = FloatVector.fromArray(SPECIES, B, i);
            va.lanewise(vectorOperator, vb).intoArray(C, i);
        }

        for (; i < C.length; i++) {
            C[i] = processSingle(A[i], B[i]);
        }

        return res;
    }
}
