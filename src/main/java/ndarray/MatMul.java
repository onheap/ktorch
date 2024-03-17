package ndarray;

import static ndarray.util.Util.SPECIES;
import static ndarray.util.Util.SPECIES_LEN;
import static ndarray.util.Util.arrOf;

import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import ndarray.util.ConcurrentUtil;
import ndarray.util.Flags;

public class MatMul {

    public static NDArray matmul(NDArray a, NDArray b) {
        if (a.shape.length != 2 || b.shape.length != 2 || a.shape[1] != b.shape[0]) {
            throw new IllegalArgumentException(
                    "MatMul only supports dense Matrix(2D-Array) right now, shapeA: %s, shapeB: %s."
                            .formatted(Arrays.toString(a.shape), Arrays.toString(b.shape)));
        }

        if (Flags.isCContiguous(a.flags) && Flags.isCContiguous(b.flags)) {
            return matmulCC(a, b);
        } else if (Flags.isCContiguous(a.flags) && Flags.isFContiguous(b.flags)) {
            return matmulCF(a, b);
        } else if (Flags.isFContiguous(a.flags) && Flags.isCContiguous(b.flags)) {
            return matmulFC(a, b);
        } else if (Flags.isFContiguous(a.flags) && Flags.isFContiguous(b.flags)) {
            return matmulFF(a, b);
        }

        throw new IllegalArgumentException("Unsupported ordering");
    }

    private static NDArray matmulCC(NDArray a, NDArray b) {
        NDArray res = NDArrays.of(arrOf(a.shape[0], b.shape[1]));

        float[] A = a.data;
        float[] B = b.data;
        float[] C = res.data;

        int ANumRows = a.shape[0], ANumCols = a.shape[1];
        int BNumRows = b.shape[0], BNumCols = b.shape[1];
        int CNumRows = ANumRows, CNumCols = BNumCols;

        ConcurrentUtil.loopFor(
                0,
                ANumRows,
                i -> {
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

    private static NDArray matmulCF(NDArray a, NDArray b) {
        NDArray res = NDArrays.of(arrOf(a.shape[0], b.shape[1]));

        float[] A = a.data;
        float[] B = b.data;
        float[] C = res.data;

        int ANumRows = a.shape[0], ANumCols = a.shape[1];
        int BNumRows = b.shape[0], BNumCols = b.shape[1];
        int CNumRows = ANumRows, CNumCols = BNumCols;

        int bound = SPECIES.loopBound(ANumCols);

        ConcurrentUtil.loopFor(
                0,
                ANumRows,
                i -> {
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

    private static NDArray matmulFC(NDArray a, NDArray b) {
        NDArray res = NDArrays.of(arrOf(a.shape[0], b.shape[1]));

        float[] A = a.data;
        float[] B = b.data;
        float[] C = res.data;

        int ANumRows = a.shape[0], ANumCols = a.shape[1];
        int BNumRows = b.shape[0], BNumCols = b.shape[1];
        int CNumRows = ANumRows, CNumCols = BNumCols;

        ConcurrentUtil.loopFor(
                0,
                ANumRows,
                i -> {
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

    private static NDArray matmulFF(NDArray a, NDArray b) {
        NDArray res = NDArrays.of(arrOf(a.shape[0], b.shape[1]));

        float[] A = a.data;
        float[] B = b.data;
        float[] C = res.data;

        int ANumRows = a.shape[0], ANumCols = a.shape[1];
        int BNumRows = b.shape[0], BNumCols = b.shape[1];
        int CNumRows = ANumRows, CNumCols = BNumCols;

        ConcurrentUtil.loopFor(
                0,
                ANumRows,
                i -> {
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
}
