package ndarray.operator;

import static ndarray.NDArrays.perform;
import static ndarray.util.Util.elementwiseOperable;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import ndarray.NDArray;
import ndarray.NDArrays;

public enum ElementWiseBinaryOperator {
    ADD(VectorOperators.ADD) {
        @Override
        public float processSingle(float a, float b) {
            return a + b;
        }
    },
    SUB(VectorOperators.SUB) {
        @Override
        public float processSingle(float a, float b) {
            return a - b;
        }
    },
    MUL(VectorOperators.MUL) {
        @Override
        public float processSingle(float a, float b) {
            return a * b;
        }
    },
    DIV(VectorOperators.DIV) {
        @Override
        public float processSingle(float a, float b) {
            return a / b;
        }
    },

    MAX(VectorOperators.MAX) {
        @Override
        public float processSingle(float a, float b) {
            return Math.max(a, b);
        }
    },

    MIN(VectorOperators.MIN) {
        @Override
        public float processSingle(float a, float b) {
            return Math.min(a, b);
        }
    };

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int SPECIES_LEN = SPECIES.length();

    public final VectorOperators.Binary vectorOperator;

    public abstract float processSingle(float a, float b);

    ElementWiseBinaryOperator(VectorOperators.Binary vectorOperator) {
        this.vectorOperator = vectorOperator;
    }

    public NDArray performBinaryOperator(NDArray a, NDArray b) {
        if (elementwiseOperable(a, b)) {
            return performElementwise(a, b);
        }

        return perform(a, b, this::processSingle);
    }

    private NDArray performElementwise(NDArray a, NDArray b) {
        NDArray res = NDArrays.zerosLike(a);

        float[] A = a.getData();
        float[] B = b.getData();
        float[] C = res.getData();

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
