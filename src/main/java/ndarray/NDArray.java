package ndarray;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.text.DecimalFormat;
import java.util.Arrays;


import static ndarray.Util.*;


@SuppressWarnings("Duplicates")
// https://github.com/tinygrad/tinygrad/blob/91a352a8e2697828a4b1eafa2bdc1a9a3b7deffa/tinygrad/tensor.py
public class NDArray {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int SPECIES_LEN = SPECIES.length();

    float[] data;
    int[] shape;
    int[] strides;
    byte flags;


    public static NDArray onesLike(NDArray other) {
        return NDArray.of(other.shape, new float[other.getSize()]);
    }

    public static NDArray of(int[] shape) {
        return new NDArray(shape);
    }

    public static NDArray of(float[] data) {
        return NDArray.of(arrOf(data.length), data);
    }

    public static NDArray of(int[] shape, float[] data) {
        return new NDArray(shape, data);
    }

    public static NDArray of(int[] shape, float[] data, Flags.Contiguous contiguous) {
        return NDArray.of(
                shape,
                contiguous.calculateStrides(shape),
                data,
                Flags.setContiguous(Flags.ZERO, contiguous));
    }

    public static NDArray of(int[] shape, int[] strides, float[] data) {
        return new NDArray(shape, strides, data);
    }


    public static NDArray of(int[] shape, int[] strides, float[] data, byte flags) {
        return new NDArray(shape, strides, data, flags);
    }


    public NDArray(int[] shape) {
        int numElements = 1;
        for (int i : shape) {
            numElements *= i;
        }

        this.data = new float[numElements];
        this.shape = shape;
        this.strides = Flags.Contiguous.C.calculateStrides(shape);
        this.flags = Flags.setContiguous(this.flags, Flags.Contiguous.C);
    }

    public NDArray(int[] shape, float[] data) {
        this.shape = shape;
        this.data = data;
        this.strides = Flags.Contiguous.C.calculateStrides(shape);
        this.flags = Flags.setContiguous(this.flags, Flags.Contiguous.C);
    }

    public NDArray(int[] shape, int[] strides, float[] data, byte flags) {
        this.shape = shape;
        this.data = data;
        this.strides = strides;
        this.flags = flags;
    }

    public NDArray(int[] shape, int[] strides, float[] data) {
        this.shape = shape;
        this.data = data;
        this.strides = strides;
        this.flags = Flags.setContiguous(this.flags, shape, strides);
    }

    public float[] getData() {
        return data;
    }

    public int[] getShape() {
        return shape;
    }

    public int[] getStrides() {
        return strides;
    }

    public int getSize() {
        return Util.getSize(shape);
    }

    public float asScalar() {
        if (shape.length != 1 || shape[0] != 1) {
            throw new IllegalArgumentException("Not a scalar");
        }
        return data[0];
    }

    public NDArray transpose() {
        return new NDArray(
                reverseArray(this.shape),
                reverseArray(this.strides),
                this.data
        );
    }

    public static int[] broadcastShapes(int[] shapeA, int[] shapeB) {
        if (shapeA.length < shapeB.length) {
            int[] temp = shapeA;
            shapeA = shapeB;
            shapeB = temp;
        }

        int diff = shapeA.length - shapeB.length;
        int len = shapeA.length;

        int[] resShape = new int[len];

        for (int i = len - 1; i >= 0; i--) {
            int elementA = shapeA[i];

            int indexB = i - diff;
            int elementB;
            if (indexB >= 0) {
                elementB = shapeB[indexB];
            } else {
                elementB = 1;
            }

            if (elementA == elementB || elementA == 1 || elementB == 1) {
                resShape[i] = Math.max(elementA, elementB);
            } else {
                throw new IllegalArgumentException(
                        "Not broadcastable shapes, shapeA: %s, shapeB: %s."
                                .formatted(Arrays.toString(shapeA), Arrays.toString(shapeB)));
            }
        }

        return resShape;
    }


    private NDArray matmulCC(NDArray other) {
        NDArray res = new NDArray(arrOf(this.shape[0], other.shape[1]));

        float[] A = this.data;
        float[] B = other.data;
        float[] C = res.data;

        int ANumRows = this.shape[0], ANumCols = this.shape[1];
        int BNumRows = other.shape[0], BNumCols = other.shape[1];
        int CNumRows = ANumRows, CNumCols = BNumCols;


        Concurrent.loopFor(0, ANumRows, i -> {
            int indexCBase = i * CNumCols;
            {
                // init the row in C
                float valA = A[i * ANumCols];
                int j = 0;
                for (; j < SPECIES.loopBound(BNumCols); j += SPECIES_LEN) {
                    var vb = FloatVector.fromArray(SPECIES, B, j);
                    vb.mul(valA).intoArray(C, indexCBase + j);
                }

                for (; j < BNumCols; j++) {
                    C[indexCBase + j] = valA * B[j];
                }
            }

            // sum up the final results
            for (int k = 1; k < BNumRows; k++) {
                int indexB = k * BNumCols;
                float valA = A[i * ANumCols + k];


                int j = 0;
                var va = FloatVector.broadcast(SPECIES, valA);
                for (; j < SPECIES.loopBound(BNumCols); j += SPECIES_LEN) {
                    var vb = FloatVector.fromArray(SPECIES, B, indexB + j);
                    var vc = FloatVector.fromArray(SPECIES, C, indexCBase + j);
                    va.fma(vb, vc).intoArray(C, indexCBase + j);
                    // vc.add(vb.mul(valA)).intoArray(C, indexCBase + j);
                }

                for (; j < BNumCols; j++) {
                    C[indexCBase + j] += valA * B[indexB + j];
                }
            }
        });

        return res;
    }

    private NDArray matmulCF(NDArray other) {
        NDArray res = new NDArray(arrOf(this.shape[0], other.shape[1]));

        float[] A = this.data;
        float[] B = other.data;
        float[] C = res.data;

        int ANumRows = this.shape[0], ANumCols = this.shape[1];
        int BNumRows = other.shape[0], BNumCols = other.shape[1];
        int CNumRows = ANumRows, CNumCols = BNumCols;

        int bound = SPECIES.loopBound(ANumCols);

        Concurrent.loopFor(0, ANumRows, i -> {
            int indexABase = i * ANumCols;
            int cIndex = i * CNumCols;

            for (int xB = 0; xB < BNumCols; xB++) {
                int indexB = xB * BNumRows;

                int j = 0;
                var sum = FloatVector.zero(SPECIES);
                for (; j < bound; j += SPECIES_LEN) {
                    var va = FloatVector.fromArray(SPECIES, A, indexABase + j);
                    var vb = FloatVector.fromArray(SPECIES, B, indexB + j);
                    sum = va.fma(vb, sum);
                }

                float total = sum.reduceLanes(VectorOperators.ADD);
                for (; j < ANumCols; j++) {
                    total += A[indexABase + j] * B[indexB + j];
                }

                C[cIndex++] = total;
            }
        });

        return res;
    }


    private NDArray matmulFC(NDArray other) {
        NDArray res = new NDArray(arrOf(this.shape[0], other.shape[1]));

        float[] A = this.data;
        float[] B = other.data;
        float[] C = res.data;

        int ANumRows = this.shape[0], ANumCols = this.shape[1];
        int BNumRows = other.shape[0], BNumCols = other.shape[1];
        int CNumRows = ANumRows, CNumCols = BNumCols;


        Concurrent.loopFor(0, ANumRows, i -> {
            int indexCBase = i * CNumCols;

            {
                // first assign R
                float valA = A[i];
                int j = 0;
                for (; j < SPECIES.loopBound(BNumCols); j += SPECIES_LEN) {
                    var vb = FloatVector.fromArray(SPECIES, B, j);
                    vb.mul(valA).intoArray(C, indexCBase + j);
                }

                for (; j < BNumCols; j++) {
                    C[indexCBase + j] = valA * B[j];
                }
            }

            // now increment it
            for (int k = 1; k < ANumCols; k++) {
                int indexB = k * BNumCols;
                float valA = A[k * ANumRows + i];

                int j = 0;
                for (; j < SPECIES.loopBound(BNumCols); j += SPECIES_LEN) {
                    var vb = FloatVector.fromArray(SPECIES, B, indexB + j);
                    var vc = FloatVector.fromArray(SPECIES, C, indexCBase + j);
                    vc.add(vb.mul(valA)).intoArray(C, indexCBase + j);
                }

                for (; j < BNumCols; j++) {
                    C[indexCBase + j] += valA * B[indexB + j];
                }
            }
        });

        return res;
    }

    private NDArray matmulFF(NDArray other) {
        NDArray res = new NDArray(arrOf(this.shape[0], other.shape[1]));

        float[] A = this.data;
        float[] B = other.data;
        float[] C = res.data;

        int ANumRows = this.shape[0], ANumCols = this.shape[1];
        int BNumRows = other.shape[0], BNumCols = other.shape[1];
        int CNumRows = ANumRows, CNumCols = BNumCols;

        Concurrent.loopFor(0, ANumRows, i -> {
            int cIndex = i * BNumCols;
            int indexB = 0;
            for (int j = 0; j < BNumCols; j++) {
                int indexA = i;
                int end = indexB + BNumRows;

                float total = 0;
                while (indexB < end) {
                    total += A[indexA] * B[indexB++];
                    indexA += ANumRows;
                }

                C[cIndex++] = total;
            }
        });

        return res;
    }

    public NDArray matmul(NDArray other) {
        if (this.shape.length != 2 || other.shape.length != 2 || this.shape[1] != other.shape[0]) {
            throw new IllegalArgumentException(
                    "MatMul only supports dense Matrix(2D-Array) right now, shapeA: %s, shapeB: %s."
                            .formatted(Arrays.toString(this.shape), Arrays.toString(other.shape)));
        }

        if (Flags.isCContiguous(this.flags) && Flags.isCContiguous(other.flags)) {
            return matmulCC(other);
        } else if (Flags.isCContiguous(this.flags) && Flags.isFContiguous(other.flags)) {
            return matmulCF(other);
        } else if (Flags.isFContiguous(this.flags) && Flags.isCContiguous(other.flags)) {
            return matmulFC(other);
        } else if (Flags.isFContiguous(this.flags) && Flags.isFContiguous(other.flags)) {
            return matmulFF(other);
        }

        throw new IllegalArgumentException("Unsupported ordering");
    }

    public NDArray sum() {
        float total = sumSubArray(data, 0, data.length);
        return NDArray.of(arrOf(total));
    }

    private float sumSubArray(float[] A, int offset, int len) {
        int i = 0;
        FloatVector sum = FloatVector.zero(SPECIES);
        for (; i < SPECIES.loopBound(len); i += SPECIES_LEN) {
            var v = FloatVector.fromArray(SPECIES, A, offset + i);
            sum = v.add(sum);
        }

        float total = sum.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            total += A[offset + i];
        }
        return total;
    }

    public NDArray sum(int axis) {
        int len = shape.length;
        if (axis < 0) {
            axis = len - axis;
        }

        if (axis >= len) {
            throw new IllegalArgumentException(
                    "axis %d is out of bounds for array of dimension %d".formatted(axis, len));
        }

        int[] newShape = delAtIndex(shape, axis);

        Flags.Contiguous contiguous = Flags.getContiguous(flags);
        if ((axis == len - 1 && contiguous == Flags.Contiguous.C)
                || (axis == 0 && contiguous == Flags.Contiguous.F)) {
            int axisLen = shape[axis];
            float[] resData = new float[Util.getSize(newShape)];
            for (int i = 0; i < resData.length; i++) {
                resData[i] = sumSubArray(data, i * axisLen, axisLen);
            }
            return NDArray.of(newShape, resData, contiguous);
        }

        NDArray res = NDArray.of(newShape);
        for (int i = 0; i < data.length; i++) {
            int[] indices = getIndices(i);
            int[] resIndices = delAtIndex(indices, axis);
            float f = this.get(indices);
            float curt = res.get(resIndices);
            res.set(resIndices, curt + f);
        }

        return res;
    }

    public NDArray mul(NDArray other) {
        elementwiseOperable(this, other);

        NDArray res = NDArray.of(shape);

        float[] A = this.data;
        float[] B = other.data;
        float[] C = res.data;

        int i = 0;
        for (; i < SPECIES.loopBound(data.length); i += SPECIES_LEN) {
            var va = FloatVector.fromArray(SPECIES, this.data, i);
            var vb = FloatVector.fromArray(SPECIES, other.data, i);
            va.mul(vb).intoArray(C, i);
        }

        for (; i < data.length; i++) {
            C[i] = A[i] * B[i];
        }

        return res;
    }


    public NDArray add(NDArray other) {
        elementwiseOperable(this, other);

        NDArray res = NDArray.of(shape);

        float[] A = this.data;
        float[] B = other.data;
        float[] C = res.data;

        int i = 0;
        for (; i < SPECIES.loopBound(data.length); i += SPECIES_LEN) {
            var va = FloatVector.fromArray(SPECIES, this.data, i);
            var vb = FloatVector.fromArray(SPECIES, other.data, i);
            va.add(vb).intoArray(C, i);
        }

        for (; i < data.length; i++) {
            C[i] = A[i] + B[i];
        }

        return res;
    }

    public NDArray performElementWiseOperator(FloatUnaryOperator op) {
        NDArray res = NDArray.of(shape);
        float[] A = this.data;
        float[] B = res.data;

        for (int i = 0; i < A.length; i++) {
            B[i] = op.applyAsFloat(A[i]);
        }

        return res;
    }


    public NDArray reshape(int[] newShape) {
        int prod = 1;
        int negIdx = -1;
        for (int i = 0; i < newShape.length; i++) {
            if (newShape[i] < 0) {
                if (negIdx != -1) {
                    throw new IllegalArgumentException("more than one negative number in shape");
                }
                negIdx = i;
            } else {
                prod *= newShape[i];
            }
        }

        int size = getSize();
        if (size != prod && (negIdx == -1 || size % prod != 0)) {
            throw new IllegalArgumentException("can not convert to new shape, size: %d, newShape: %s".formatted(size, Arrays.toString(newShape)));
        }

        if (negIdx != -1) {
            newShape = Arrays.copyOf(newShape, newShape.length);
            newShape[negIdx] = size / prod;
        }

        return new NDArray(newShape, this.data);
    }

    private int getFlatIndex(int[] indices) {
        return Util.getFlatIndex(indices, strides);
    }

    private int[] getIndices(int flatIndex) {
        return Util.getIndices(flatIndex, shape);
    }

    public float get(int[] indices) {
        return data[getFlatIndex(indices)];
    }

    public void set(int[] indices, float v) {
        data[getFlatIndex(indices)] = v;
    }

    @Override
    public String toString() {
        final var df = new DecimalFormat("##.####");
        String a = info();
        int len = shape.length;

        a += "[".repeat(len);

        for (int i = 0; i < data.length; i++) {
            int[] indices = getIndices(i);

            if (i != 0) {
                int zeros = 0;
                for (int j = len - 1; j >= 0; j--) {
                    if (indices[j] == 0) {
                        zeros++;
                    } else {
                        break;
                    }
                }

                if (zeros > 0) {
                    a += "]".repeat(zeros);
                    a += "\n";
                    a += " ".repeat(len - zeros);
                    a += "[".repeat(zeros);
                } else {
                    a += ", ";
                }
            }

            var f = get(indices);
            a += df.format(f);
        }

        a += "]".repeat(len);

        return a;
    }

    public String info() {
        return """
                Info:
                 Shape: %s
                 Strides: %s
                 Contiguous: %s
                """.formatted(
                Arrays.toString(shape),
                Arrays.toString(strides),
                Flags.getContiguous(flags));
    }

    public static void main(String[] args) {
        var a = NDArray.of(arrOf(4, 3), arrOfF(
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        ));
        System.out.println(a);

        NDArray b = a.sum(0);
        System.out.println(b);
    }

    private static void testReshape() {
        var a = NDArray.of(arrOf(3, 2), arrOfF(
                1, 4,
                2, 5,
                3, 6
        ));

        NDArray b = a.reshape(arrOf(-1, 6));
        System.out.println(b);
    }

    private static void testMatmulFC() {
        var a = new NDArray(arrOf(3, 2), arrOfF(
                1, 4,
                2, 5,
                3, 6
        ));

        var b = new NDArray(arrOf(3, 2), arrOfF(
                1, 2,
                3, 4,
                5, 6
        ));

        var c = a.transpose();
        System.out.println(c);

        NDArray res = c.matmul(b);
        System.out.println(res);
    }

    private static void testMatmulFF() {
        var a = new NDArray(arrOf(3, 2), arrOfF(
                1, 4,
                2, 5,
                3, 6
        ));

        var b = new NDArray(arrOf(2, 3), arrOfF(
                1, 3, 5,
                2, 4, 6
        ));

        var c = a.transpose();
        var d = b.transpose();

        System.out.println(c);
        System.out.println(d);

        NDArray res = c.matmul(d);

        System.out.println(res);
    }


    private static void testMatmulCF() {
        var a = new NDArray(arrOf(2, 3), arrOfF(
                1, 2, 3,
                4, 5, 6
        ));

        var b = new NDArray(arrOf(2, 3), arrOfF(
                1, 3, 5,
                2, 4, 6
        ));

        var c = b.transpose();

        System.out.println(c);

        NDArray d = a.matmul(c);

        System.out.println(d);
    }

    private static void testPrint() {
        float[] data = arrOf(0.0F, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        int[] shape = arrOf(4, 1, 3);

        NDArray a = new NDArray(shape, data);

        System.out.println(a);
    }

    private static void testBroadcast() {
        int[] res = NDArray.broadcastShapes(arrOf(8, 1, 6, 1), arrOf(7, 1, 5));
        System.out.println(Arrays.toString(res));
    }


    private static void testMatmulCC() {
        var a = new NDArray(arrOf(2, 3), arrOfF(
                1, 2, 3,
                4, 5, 6
        ));

        var b = new NDArray(arrOf(3, 2), arrOfF(
                1, 2,
                3, 4,
                5, 6
        ));

        NDArray res = a.matmul(b);

        System.out.println(res);
    }

    private static void testTranspose() {
        var a = new NDArray(arrOf(4, 3), arrOfF(
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
                10, 11, 12
        ));

        System.out.println(a);
        NDArray b = a.transpose();
        System.out.println(b);
    }

    private static void testTranspose1() {
        var a = new NDArray(arrOf(2, 2, 3), arrOfF(
                1, 2, 3,
                4, 5, 6,

                7, 8, 9,
                10, 11, 12
        ));

        System.out.println(a);

        var b = a.transpose();
        System.out.println(b);
    }
}