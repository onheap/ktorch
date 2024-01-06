package ndarray;

import static ndarray.ShapeUtil.*;
import static ndarray.Util.*;

import java.util.Arrays;
import java.util.Iterator;

public class NDArrays {

    // copy shape, strides, contiguous
    public static NDArray onesLike(NDArray other) {
        float[] data = new float[other.getSize()];
        Arrays.fill(data, 1);
        return NDArrays.of(other.shape, data, Flags.getContiguous(other.flags));
    }

    public static NDArray zerosLike(NDArray other) {
        return NDArrays.of(other.shape, Flags.getContiguous(other.flags));
    }

    public static NDArray arange(int start, int end) {
        float[] data = new float[end - start];
        for (int i = 0; i < end - start; i++) {
            data[i] = start + i;
        }

        return NDArrays.of(data);
    }

    public static NDArray of(int[] shape) {
        return NDArrays.of(shape, Flags.Contiguous.C);
    }

    public static NDArray fill(int[] shape, float v) {
        float[] data = new float[ShapeUtil.getSize(shape)];
        Arrays.fill(data, v);
        return NDArrays.of(shape, data, Flags.Contiguous.C);
    }

    public static NDArray of(float[] data) {
        return NDArrays.of(arrOf(data.length), data);
    }

    public static NDArray of(float[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            throw new IllegalArgumentException("not a valid matrix");
        }

        int m = matrix.length;
        int n = matrix[0].length;

        float[] data = new float[m * n];

        for (int i = 0; i < m; i++) {
            if (matrix[i].length != n) {
                throw new IllegalArgumentException("not a valid matrix");
            }

            System.arraycopy(matrix[i], 0, data, i * n, n);
        }

        return NDArrays.of(arrOf(m, n), data);
    }

    public static NDArray ofScalar(float data) {
        return NDArrays.of(new int[0], arrOf(data));
    }

    public static NDArray of(int[] shape, float[] data) {
        return NDArrays.of(shape, data, Flags.Contiguous.C);
    }

    public static NDArray of(int[] shape, Flags.Contiguous contiguous) {
        return NDArrays.of(
                shape,
                contiguous.calculateStrides(shape),
                new float[ShapeUtil.getSize(shape)],
                Flags.setContiguous(Flags.ZERO, contiguous));
    }

    public static NDArray of(int[] shape, float[] data, Flags.Contiguous contiguous) {
        return NDArrays.of(
                shape,
                contiguous.calculateStrides(shape),
                data,
                Flags.setContiguous(Flags.ZERO, contiguous));
    }

    public static NDArray of(int[] shape, int[] strides, float[] data) {
        return NDArrays.of(shape, strides, data, Flags.setContiguous(Flags.ZERO, shape, strides));
    }

    public static NDArray of(int[] shape, int[] strides, float[] data, byte flags) {
        return new NDArray(shape, strides, data, flags);
    }

    public static NDArray stack(NDArray... ndArrays) {
        if (ndArrays.length == 0 || ndArrays[0].getContiguous() != Flags.Contiguous.C) {
            throw new IllegalArgumentException("can not merge NDArrays");
        }

        int len = ndArrays.length;

        NDArray first = ndArrays[0];
        int size = first.getSize();
        float[] data = new float[len * size];

        for (int i = 0; i < len; i++) {
            NDArray curt = ndArrays[i];
            assert elementwiseOperable(first, curt);

            System.arraycopy(curt.data, 0, data, i * size, size);
        }

        int[] newShape = new int[first.shape.length + 1];
        newShape[0] = len;
        System.arraycopy(first.shape, 0, newShape, 1, first.shape.length);

        return NDArrays.of(newShape, data);
    }

    public static NDArray perform(NDArray a, NDArray b, FloatBinaryOperator op) {
        if (shapesEqual(a, b)) {
            return performIteratively(a, b, op);
        }

        return performBroadcastly(a, b, op);
    }

    protected static NDArray performIteratively(NDArray a, NDArray b, FloatBinaryOperator op) {
        assertShapesEqual(a, b);
        NDArray res = NDArrays.of(a.shape);
        Iterator<Float> A = a.iterator();
        Iterator<Float> B = b.iterator();
        float[] output = res.data;

        for (int i = 0; i < output.length; i++) {
            float va = A.next();
            float vb = B.next();
            output[i] = op.applyAsFloat(va, vb);
        }
        return res;
    }

    protected static NDArray performBroadcastly(NDArray a, NDArray b, FloatBinaryOperator op) {
        boolean scalarA = a.isScalar();
        boolean scalarB = b.isScalar();
        if (scalarA || scalarB) {
            if (scalarA && scalarB) {
                return NDArrays.ofScalar(op.applyAsFloat(a.asScalar(), b.asScalar()));
            }

            if (scalarA) {
                return performScalarWith(a, b, op);
            }

            return performWithScalar(a, b, op);
        }

        NDArray res = NDArrays.of(broadcastShapes(a.shape, b.shape));
        int[] shapeA = a.shape;
        int[] shapeB = b.shape;
        int[] shapeC = res.shape;

        int lenA = shapeA.length;
        int lenB = shapeB.length;
        int lenC = shapeC.length;

        int[] indicesA = new int[lenA];
        int[] indicesB = new int[lenB];
        int[] indicesC = new int[lenC];

        do {
            System.arraycopy(indicesC, lenC - lenA, indicesA, 0, lenA);
            System.arraycopy(indicesC, lenC - lenB, indicesB, 0, lenB);

            ShapeUtil.constrainIndices(indicesA, shapeA);
            ShapeUtil.constrainIndices(indicesB, shapeB);

            res.set(indicesC, op.applyAsFloat(a.get(indicesA), b.get(indicesB)));
        } while (ShapeUtil.increaseIndices(indicesC, shapeC));

        return res;
    }

    protected static NDArray performScalarWith(NDArray a, NDArray b, FloatBinaryOperator op) {
        float va = a.asScalar();
        NDArray res = NDArrays.of(a.shape);

        Iterator<Float> B = b.iterator();
        float[] output = res.data;

        for (int i = 0; i < output.length; i++) {
            float vb = B.next();
            output[i] = op.applyAsFloat(va, vb);
        }
        return res;
    }

    protected static NDArray performWithScalar(NDArray a, NDArray b, FloatBinaryOperator op) {
        float vb = b.asScalar();
        NDArray res = NDArrays.of(a.shape);

        Iterator<Float> A = a.iterator();
        float[] output = res.data;

        for (int i = 0; i < output.length; i++) {
            float va = A.next();
            output[i] = op.applyAsFloat(va, vb);
        }
        return res;
    }
}
