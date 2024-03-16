package ndarray.operator;

import static ndarray.utils.ShapeUtil.copyIndices;
import static ndarray.utils.ShapeUtil.reduceShape;
import static ndarray.utils.Util.SPECIES;
import static ndarray.utils.Util.SPECIES_LEN;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import ndarray.Flags;
import ndarray.NDArray;
import ndarray.NDArrays;
import ndarray.utils.ShapeUtil;

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

    public NDArray reduceAlongDimension(NDArray ndArray, int dim, boolean keepDims) {
        int[] shape = ndArray.getShape();
        int len = shape.length;
        dim = dim < 0 ? len + dim : dim;

        if (dim < 0 || dim >= len) {
            throw new IllegalArgumentException(
                    "dim %d is out of bounds for array of dimension %d".formatted(dim, len));
        }

        int[] newShape = reduceShape(shape, dim, keepDims);

        Flags.Contiguous contiguous = ndArray.getContiguous();
        if ((dim == len - 1 && contiguous == Flags.Contiguous.C)
                || (dim == 0 && contiguous == Flags.Contiguous.F)) {
            int axisLen = shape[dim];
            float[] resData = new float[ShapeUtil.getSize(newShape)];
            for (int i = 0; i < resData.length; i++) {
                resData[i] = elementWiseReduce(ndArray.getData(), i * axisLen, axisLen);
            }
            return NDArrays.of(newShape, resData, contiguous);
        }

        NDArray res = NDArrays.fill(newShape, getInit());
        int[] resIndices = new int[res.getShape().length];
        for (int[] indices : ndArray.indices()) {
            copyIndices(indices, resIndices, dim, keepDims);

            float f = ndArray.get(indices);
            float curt = res.get(resIndices);
            res.set(resIndices, processSingle(curt, f));
        }

        return res;
    }

    public float elementWiseReduce(float[] A, int offset, int len) {
        int i = 0;
        FloatVector temp = FloatVector.broadcast(SPECIES, getInit());
        for (; i < SPECIES.loopBound(len); i += SPECIES_LEN) {
            var v = FloatVector.fromArray(SPECIES, A, offset + i);
            temp = v.lanewise(vectorOperator, temp);
        }

        float res = temp.reduceLanes(vectorOperator);
        for (; i < len; i++) {
            res = processSingle(A[offset + i], res);
        }
        return res;
    }
}
