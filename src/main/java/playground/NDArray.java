package playground;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import javax.xml.stream.FactoryConfigurationError;
import java.text.DecimalFormat;
import java.util.Arrays;


import static playground.Util.*;


@SuppressWarnings("Duplicates")
// https://github.com/tinygrad/tinygrad/blob/91a352a8e2697828a4b1eafa2bdc1a9a3b7deffa/tinygrad/tensor.py
public class NDArray {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int SPECIES_LEN = SPECIES.length();

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
        this.flags = Flags.setCOrder(this.flags);
    }

    public NDArray(int[] shape, float[] data) {
        this.shape = shape;
        this.data = data;
        this.strides = calculateStrides(shape);
        this.flags = Flags.setCOrder(this.flags);
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
        this.flags = Flags.setOrder(this.flags, shape, strides);
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

    public NDArray transpose() {
        return new NDArray(
                reverseArray(this.shape),
                reverseArray(this.strides),
                this.data
        );
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

        if (Flags.isCOrder(this.flags) && Flags.isCOrder(other.flags)) {
            return matmulCC(other);
        } else if (Flags.isCOrder(this.flags) && Flags.isFOrder(other.flags)) {
            return matmulCF(other);
        } else if (Flags.isFOrder(this.flags) && Flags.isCOrder(other.flags)) {
            return matmulFC(other);
        } else if (Flags.isFOrder(this.flags) && Flags.isFOrder(other.flags)) {
            return matmulFF(other);
        }

        throw new IllegalArgumentException("Unsupported ordering");
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
        String a = info();
        int len = shape.length;

        a += "[".repeat(len);

        int[] indices = new int[len];

        for (int i = 0; i < data.length; i++) {
            int temp = i;
            for (int j = len - 1; j >= 0; j--) {
                indices[j] = temp % shape[j];
                temp /= shape[j];
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

    public String info() {
        return """
                Info:
                 Shape: %s
                 Strides: %s
                 IsCOrder: %b
                 IsFOrder: %b
                """.formatted(
                Arrays.toString(shape),
                Arrays.toString(strides),
                Flags.isCOrder(flags),
                Flags.isFOrder(flags));
    }

    public static void main(String[] args) {
        testMatmulFF();
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