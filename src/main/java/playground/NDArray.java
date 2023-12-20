package playground;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.text.DecimalFormat;
import java.util.Arrays;


import static playground.Util.arrOf;
import static playground.Util.arrOfF;


// https://github.com/tinygrad/tinygrad/blob/91a352a8e2697828a4b1eafa2bdc1a9a3b7deffa/tinygrad/tensor.py
public class NDArray {

    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private static final byte FLAG_ORDER_MASK = 0b11;
    private static final byte FLAG_C_ORDER = 0b01;
    private static final byte FLAG_F_ORDER = 0b10;
    private static final byte FLAG_S_ORDER = 0b11;

    float[] data;
    int[] shape;
    int[] strides;
    byte flags;


    public NDArray(int[] shape) {
        int numElements = 1;
        for (int i : shape) {
            numElements *= i;
        }

        this.data = new float[numElements];
        this.shape = shape;
        this.strides = calculateStrides(shape);
        this.setCOrder();
    }

    public NDArray(int[] shape, float[] data) {
        this.shape = shape;
        this.data = data;
        this.strides = calculateStrides(shape);
        this.setCOrder();
    }

    public float[] getData() {
        return data;
    }

    private void setCOrder() {
        this.flags &= ~FLAG_ORDER_MASK;
        this.flags |= FLAG_C_ORDER;
    }

    private void setFOrder() {
        this.flags &= ~FLAG_ORDER_MASK;
        this.flags |= FLAG_F_ORDER;
    }

    private void setSOrder() {
        this.flags &= ~FLAG_ORDER_MASK;
        this.flags |= FLAG_S_ORDER;
    }

    public boolean isCOrder() {
        return (this.flags & FLAG_ORDER_MASK) == FLAG_C_ORDER;
    }

    public boolean isFOrder() {
        return (this.flags & FLAG_ORDER_MASK) == FLAG_F_ORDER;
    }

    private int[] calculateStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
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

    public NDArray matMulSimple(NDArray other) {
        if (this.shape.length != 2 || other.shape.length != 2 || this.shape[1] != other.shape[0]) {
            throw new IllegalArgumentException(
                    "MatMul only supports dense Matrix(2D-Array) right now, shapeA: %s, shapeB: %s."
                            .formatted(Arrays.toString(this.shape), Arrays.toString(other.shape)));
        }

        NDArray res = new NDArray(arrOf(this.shape[0], other.shape[1]));

        float[] A = this.data;
        float[] B = other.data;
        float[] C = res.data;

        int ANumRows = this.shape[0], ANumCols = this.shape[1];
        int BNumRows = other.shape[0], BNumCols = other.shape[1];
        int CNumRows = ANumRows, CNumCols = BNumCols;

        for (int i = 0; i < ANumRows; i++) {
            int indexCBase = i * CNumCols;
            {
                // init the row in C
                float valA = A[i * ANumCols];
                for (int j = 0; j < BNumCols; j++) {
                    C[indexCBase + j] = valA * B[j];
                }
            }

            // sum up the final results
            for (int k = 1; k < BNumRows; k++) {
                int indexC = indexCBase;
                int indexB = k * BNumCols;

                float valA = A[i * ANumCols + k];
                for (int j = 0; j < BNumCols; j++) {
                    C[indexC++] += valA * B[indexB++];
                }
            }
        }

        return res;
    }

    public NDArray matMulVector(NDArray other) {
        if (this.shape.length != 2 || other.shape.length != 2 || this.shape[1] != other.shape[0]) {
            throw new IllegalArgumentException(
                    "MatMul only supports dense Matrix(2D-Array) right now, shapeA: %s, shapeB: %s."
                            .formatted(Arrays.toString(this.shape), Arrays.toString(other.shape)));
        }

        NDArray res = new NDArray(arrOf(this.shape[0], other.shape[1]));

        float[] A = this.data;
        float[] B = other.data;
        float[] C = res.data;

        int ANumRows = this.shape[0], ANumCols = this.shape[1];
        int BNumRows = other.shape[0], BNumCols = other.shape[1];
        int CNumRows = ANumRows, CNumCols = BNumCols;

        for (int i = 0; i < ANumRows; i++) {
            int indexCBase = i * CNumCols;
            {
                // init the row in C
                float valA = A[i * ANumCols];
                int j;
                for (j = 0; j < SPECIES.loopBound(BNumCols); j += SPECIES.length()) {
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


                int j;
                for (j = 0; j < SPECIES.loopBound(BNumCols); j += SPECIES.length()) {
                    var vb = FloatVector.fromArray(SPECIES, B, indexB + j);
                    var vc = FloatVector.fromArray(SPECIES, C, indexCBase + j);
                    vc.add(vb.mul(valA)).intoArray(C, indexCBase + j);
                }

                for (; j < BNumCols; j++) {
                    C[indexCBase + j] += valA * B[indexB + j];
                }
            }
        }

        return res;
    }


    public NDArray matMulVectorConcurrent(NDArray other) {
        if (this.shape.length != 2 || other.shape.length != 2 || this.shape[1] != other.shape[0]) {
            throw new IllegalArgumentException(
                    "MatMul only supports dense Matrix(2D-Array) right now, shapeA: %s, shapeB: %s."
                            .formatted(Arrays.toString(this.shape), Arrays.toString(other.shape)));
        }

        NDArray res = new NDArray(arrOf(this.shape[0], other.shape[1]));

        float[] A = this.data;
        float[] B = other.data;
        float[] C = res.data;

        int ANumRows = this.shape[0], ANumCols = this.shape[1];
        int BNumRows = other.shape[0], BNumCols = other.shape[1];
        int CNumRows = ANumRows, CNumCols = BNumCols;


        Util.concurrentLoopFor(0, ANumRows, i -> {
            int indexCBase = i * CNumCols;
            {
                // init the row in C
                float valA = A[i * ANumCols];
                int j;
                for (j = 0; j < SPECIES.loopBound(BNumCols); j += SPECIES.length()) {
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


                int j;
                for (j = 0; j < SPECIES.loopBound(BNumCols); j += SPECIES.length()) {
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


    private int getFlatIndex(int[] indices) {
        int index = 0;
        for (int i = 0; i < indices.length; i++) {
            index += indices[i] * strides[i];
        }
        return index;
    }

    public float get(int[] indices) {
        return data[getFlatIndex(indices)];
    }

    @Override
    public String toString() {
        final var df = new DecimalFormat("##.####");

        // TODO: special visualization for 1,2,3 dimension arrays

        String a = "";
        int len = shape.length;

        a += "[".repeat(len);

        int[] indices = new int[len];

        for (int i = 0; i < data.length; i++) {
            int temp = i;
            for (int j = 0; j < len; j++) {
                indices[j] = temp / strides[j];
                temp = temp % strides[j];
            }

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

    public static void main(String[] args) {
        testMatMulVector();
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

    private static void testMatMulSimple() {
        var a = new NDArray(arrOf(2, 3), arrOfF(
                1, 2, 3,
                4, 5, 6
        ));

        var b = new NDArray(arrOf(3, 2), arrOfF(
                1, 2,
                3, 4,
                5, 6
        ));

        NDArray res = a.matMulSimple(b);

        System.out.println(res);
    }

    private static void testMatMulVector() {
        var a = new NDArray(arrOf(2, 3), arrOfF(
                1, 2, 3,
                4, 5, 6
        ));

        var b = new NDArray(arrOf(3, 2), arrOfF(
                1, 2,
                3, 4,
                5, 6
        ));

        NDArray res = a.matMulVector(b);

        System.out.println(res);
    }
}