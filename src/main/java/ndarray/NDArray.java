package ndarray;

import static ndarray.Util.*;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

@SuppressWarnings("Duplicates")
// There are lots of duplicate code that can be optimized on readability wise. However,
// the Benchmark result of extracting common code fragments into functions is not good.
public class NDArray implements Iterable<Float> {

    public static final NDArray EMPTY = NDArray.of(new float[0]);

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int SPECIES_LEN = SPECIES.length();

    final float[] data;
    final int[] shape;
    final int[] strides;
    final byte flags;

    // copy shape, strides, contiguous
    public static NDArray onesLike(NDArray other) {
        float[] data = new float[other.getSize()];
        Arrays.fill(data, 1);
        return NDArray.of(other.shape, data, Flags.getContiguous(other.flags));
    }

    public static NDArray zerosLike(NDArray other) {
        return NDArray.of(other.shape, Flags.getContiguous(other.flags));
    }

    public static NDArray of(int[] shape) {
        return NDArray.of(shape, Flags.Contiguous.C);
    }

    public static NDArray fill(int[] shape, float v) {
        float[] data = new float[Util.getSize(shape)];
        Arrays.fill(data, v);
        return NDArray.of(shape, data, Flags.Contiguous.C);
    }

    public static NDArray of(float[] data) {
        return NDArray.of(arrOf(data.length), data);
    }

    public static NDArray of(int[] shape, float[] data) {
        return NDArray.of(shape, data, Flags.Contiguous.C);
    }

    public static NDArray of(int[] shape, Flags.Contiguous contiguous) {
        return NDArray.of(
                shape,
                contiguous.calculateStrides(shape),
                new float[Util.getSize(shape)],
                Flags.setContiguous(Flags.ZERO, contiguous));
    }

    public static NDArray of(int[] shape, float[] data, Flags.Contiguous contiguous) {
        return NDArray.of(
                shape,
                contiguous.calculateStrides(shape),
                data,
                Flags.setContiguous(Flags.ZERO, contiguous));
    }

    public static NDArray of(int[] shape, int[] strides, float[] data) {
        return NDArray.of(shape, strides, data, Flags.setContiguous(Flags.ZERO, shape, strides));
    }

    public static NDArray of(int[] shape, int[] strides, float[] data, byte flags) {
        return new NDArray(shape, strides, data, flags);
    }

    public NDArray(int[] shape, float[] data) {
        this.shape = shape;
        this.data = data;
        this.strides = Flags.Contiguous.C.calculateStrides(shape);
        this.flags = Flags.setContiguous(Flags.ZERO, Flags.Contiguous.C);
    }

    public NDArray(int[] shape, int[] strides, float[] data, byte flags) {
        this.shape = shape;
        this.data = data;
        this.strides = strides;
        this.flags = flags;
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

    public Iterator<Float> iterator() {
        return new Iterator<>() {
            private int curt = 0;
            private final int size = getSize();
            private final boolean isCContiguous = getContiguous() == Flags.Contiguous.C;

            private final int[] indices = isCContiguous ? null : new int[shape.length];

            @Override
            public boolean hasNext() {
                return curt < size;
            }

            @Override
            public Float next() {
                if (curt >= size) {
                    throw new NoSuchElementException(
                            "iterator reached the end, curt: %d, size: %d".formatted(curt, size));
                }

                if (isCContiguous) {
                    return data[curt++];
                } else {
                    return get(calIndices(curt++, shape, indices));
                }
            }
        };
    }

    public Iterator<int[]> indicesIterator() {
        return new Iterator<>() {
            private int curt = 0;
            private final int size = getSize();

            @Override
            public boolean hasNext() {
                return curt < size;
            }

            @Override
            public int[] next() {
                if (curt >= size) {
                    throw new NoSuchElementException(
                            "iterator reached the end, curt: %d, size: %d".formatted(curt, size));
                }

                return getIndices(curt++);
            }
        };
    }

    public Flags.Contiguous getContiguous() {
        return Flags.getContiguous(flags);
    }

    public float asScalar() {
        if (shape.length != 1 || shape[0] != 1) {
            throw new IllegalArgumentException("Not a scalar");
        }
        return data[0];
    }

    public NDArray transpose() {
        return NDArray.of(reverseArray(this.shape), reverseArray(this.strides), this.data);
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
        NDArray res = NDArray.of(arrOf(this.shape[0], other.shape[1]));

        float[] A = this.data;
        float[] B = other.data;
        float[] C = res.data;

        int ANumRows = this.shape[0], ANumCols = this.shape[1];
        int BNumRows = other.shape[0], BNumCols = other.shape[1];
        int CNumRows = ANumRows, CNumCols = BNumCols;

        Concurrent.loopFor(
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

    private NDArray matmulCF(NDArray other) {
        NDArray res = NDArray.of(arrOf(this.shape[0], other.shape[1]));

        float[] A = this.data;
        float[] B = other.data;
        float[] C = res.data;

        int ANumRows = this.shape[0], ANumCols = this.shape[1];
        int BNumRows = other.shape[0], BNumCols = other.shape[1];
        int CNumRows = ANumRows, CNumCols = BNumCols;

        int bound = SPECIES.loopBound(ANumCols);

        Concurrent.loopFor(
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

    private NDArray matmulFC(NDArray other) {
        NDArray res = NDArray.of(arrOf(this.shape[0], other.shape[1]));

        float[] A = this.data;
        float[] B = other.data;
        float[] C = res.data;

        int ANumRows = this.shape[0], ANumCols = this.shape[1];
        int BNumRows = other.shape[0], BNumCols = other.shape[1];
        int CNumRows = ANumRows, CNumCols = BNumCols;

        Concurrent.loopFor(
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

    private NDArray matmulFF(NDArray other) {
        NDArray res = NDArray.of(arrOf(this.shape[0], other.shape[1]));

        float[] A = this.data;
        float[] B = other.data;
        float[] C = res.data;

        int ANumRows = this.shape[0], ANumCols = this.shape[1];
        int BNumRows = other.shape[0], BNumCols = other.shape[1];
        int CNumRows = ANumRows, CNumCols = BNumCols;

        Concurrent.loopFor(
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

    public NDArray maximum(float v) {
        assert elementwiseOperable(this);

        NDArray res = NDArray.zerosLike(this);

        float[] A = this.data;
        float[] B = res.data;

        int i = 0;
        int len = A.length;
        FloatVector max = FloatVector.broadcast(SPECIES, v);
        for (; i < SPECIES.loopBound(len); i += SPECIES_LEN) {
            var va = FloatVector.fromArray(SPECIES, A, i);
            va.max(max).intoArray(B, i);
        }

        for (; i < len; i++) {
            B[i] = Math.max(A[i], v);
        }

        return res;
    }

    public NDArray maximum(NDArray other) {
        assertShapesEqual(this, other);

        if (elementwiseOperable(this, other)) {
            NDArray res = NDArray.zerosLike(this);

            float[] A = this.data;
            float[] B = other.data;
            float[] C = res.data;

            int i = 0;
            int len = A.length;
            for (; i < SPECIES.loopBound(len); i += SPECIES_LEN) {
                var va = FloatVector.fromArray(SPECIES, A, i);
                var vb = FloatVector.fromArray(SPECIES, B, i);
                va.max(vb).intoArray(C, i);
            }

            for (; i < len; i++) {
                C[i] = Math.max(A[i], B[i]);
            }

            return res;
        }

        return NDArray.of(shape, iterateToFloatArray(this, other, Math::max));
    }

    public NDArray minimum(float v) {
        assert elementwiseOperable(this);

        NDArray res = NDArray.zerosLike(this);

        float[] A = this.data;
        float[] B = res.data;

        int i = 0;
        int len = A.length;
        FloatVector max = FloatVector.broadcast(SPECIES, v);
        for (; i < SPECIES.loopBound(len); i += SPECIES_LEN) {
            var va = FloatVector.fromArray(SPECIES, A, i);
            va.max(max).intoArray(B, i);
        }

        for (; i < len; i++) {
            B[i] = Math.min(A[i], v);
        }

        return res;
    }

    public NDArray minimum(NDArray other) {
        assertShapesEqual(this, other);

        if (elementwiseOperable(this, other)) {
            NDArray res = NDArray.zerosLike(this);

            float[] A = this.data;
            float[] B = other.data;
            float[] C = res.data;

            int i = 0;
            int len = A.length;
            for (; i < SPECIES.loopBound(len); i += SPECIES_LEN) {
                var va = FloatVector.fromArray(SPECIES, A, i);
                var vb = FloatVector.fromArray(SPECIES, B, i);
                va.min(vb).intoArray(C, i);
            }

            for (; i < len; i++) {
                C[i] = Math.min(A[i], B[i]);
            }

            return res;
        }

        return NDArray.of(shape, iterateToFloatArray(this, other, Math::min));
    }

    public NDArray sum() {
        float total = elementWiseReduce(data, 0, data.length, ElementWiseReduceOperator.SUM);
        return NDArray.of(arrOf(total));
    }

    public NDArray sum(int axis) {
        return reduceAlongAxis(axis, ElementWiseReduceOperator.SUM);
    }

    public NDArray max() {
        float max = elementWiseReduce(data, 0, data.length, ElementWiseReduceOperator.MAX);
        return NDArray.of(arrOf(max));
    }

    public NDArray max(int axis) {
        return reduceAlongAxis(axis, ElementWiseReduceOperator.MAX);
    }

    private NDArray reduceAlongAxis(int axis, ElementWiseReduceOperator op) {
        int len = shape.length;
        if (axis < 0) {
            axis = len - axis;
        }

        if (axis >= len) {
            throw new IllegalArgumentException(
                    "axis %d is out of bounds for array of dimension %d".formatted(axis, len));
        }

        int[] newShape = delAtIndex(shape, axis);
        Flags.Contiguous contiguous = getContiguous();
        if ((axis == len - 1 && contiguous == Flags.Contiguous.C)
                || (axis == 0 && contiguous == Flags.Contiguous.F)) {
            int axisLen = shape[axis];
            float[] resData = new float[Util.getSize(newShape)];
            for (int i = 0; i < resData.length; i++) {
                resData[i] = elementWiseReduce(data, i * axisLen, axisLen, op);
            }
            return NDArray.of(newShape, resData, contiguous);
        }

        NDArray res = NDArray.fill(newShape, op.getInit());

        int[] indices = new int[shape.length];
        for (int i = 0; i < data.length; i++) {
            indices = Util.calIndices(i, shape, indices);
            int[] resIndices = delAtIndex(indices, axis);
            float f = this.get(indices);
            float curt = res.get(resIndices);
            res.set(resIndices, op.processSingle(curt, f));
        }

        return res;
    }

    private float elementWiseReduce(float[] A, int offset, int len, ElementWiseReduceOperator op) {
        int i = 0;
        FloatVector temp = FloatVector.broadcast(SPECIES, op.getInit());
        for (; i < SPECIES.loopBound(len); i += SPECIES_LEN) {
            var v = FloatVector.fromArray(SPECIES, A, offset + i);
            temp = v.lanewise(op.vectorOperator, temp);
        }

        float res = temp.reduceLanes(op.vectorOperator);
        for (; i < len; i++) {
            res = op.processSingle(A[offset + i], res);
        }
        return res;
    }

    public NDArray add(NDArray other) {
        assertShapesEqual(this, other);

        if (elementwiseOperable(this, other)) {
            NDArray res = NDArray.zerosLike(this);

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

        return NDArray.of(shape, iterateToFloatArray(this, other, (a, b) -> a + b));
    }

    public NDArray sub(NDArray other) {
        assertShapesEqual(this, other);

        if (elementwiseOperable(this, other)) {
            NDArray res = NDArray.zerosLike(this);

            float[] A = this.data;
            float[] B = other.data;
            float[] C = res.data;

            int i = 0;
            for (; i < SPECIES.loopBound(data.length); i += SPECIES_LEN) {
                var va = FloatVector.fromArray(SPECIES, this.data, i);
                var vb = FloatVector.fromArray(SPECIES, other.data, i);
                va.sub(vb).intoArray(C, i);
            }

            for (; i < data.length; i++) {
                C[i] = A[i] - B[i];
            }

            return res;
        }

        return NDArray.of(shape, iterateToFloatArray(this, other, (a, b) -> a - b));
    }

    public NDArray mul(NDArray other) {
        assertShapesEqual(this, other);

        if (elementwiseOperable(this, other)) {
            NDArray res = NDArray.zerosLike(this);

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

        return NDArray.of(shape, iterateToFloatArray(this, other, (a, b) -> a * b));
    }

    public NDArray div(NDArray other) {
        assertShapesEqual(this, other);
        if (elementwiseOperable(this, other)) {
            NDArray res = NDArray.zerosLike(this);

            float[] A = this.data;
            float[] B = other.data;
            float[] C = res.data;

            int i = 0;
            for (; i < SPECIES.loopBound(data.length); i += SPECIES_LEN) {
                var va = FloatVector.fromArray(SPECIES, this.data, i);
                var vb = FloatVector.fromArray(SPECIES, other.data, i);
                va.div(vb).intoArray(C, i);
            }

            for (; i < data.length; i++) {
                C[i] = A[i] / B[i];
            }

            return res;
        }

        return NDArray.of(shape, iterateToFloatArray(this, other, (a, b) -> a / b));
    }

    public NDArray log() {
        return performElementWiseOperator(ElementWiseUnaryOperator.LOG);
    }

    public NDArray exp() {
        return performElementWiseOperator(ElementWiseUnaryOperator.EXP);
    }

    private float[] iterateToFloatArray(NDArray a, NDArray b, FloatBinaryOperator op) {
        Iterator<Float> A = a.iterator();
        Iterator<Float> B = b.iterator();
        float[] output = new float[a.getSize()];

        for (int i = 0; i < output.length; i++) {
            float va = A.next();
            float vb = B.next();
            output[i] = op.applyAsFloat(va, vb);
        }
        return output;
    }

    private float[] iterateToFloatArray(NDArray a, FloatUnaryOperator op) {
        Iterator<Float> A = a.iterator();
        float[] output = new float[a.getSize()];

        for (int i = 0; i < output.length; i++) {
            float va = A.next();
            output[i] = op.applyAsFloat(va);
        }
        return output;
    }

    public NDArray performElementWiseOperator(FloatUnaryOperator op) {
        assert elementwiseOperable(this);
        NDArray res = NDArray.zerosLike(this);
        float[] A = this.data;
        float[] B = res.data;

        for (int i = 0; i < A.length; i++) {
            B[i] = op.applyAsFloat(A[i]);
        }

        return res;
    }

    public NDArray performElementWiseOperator(ElementWiseUnaryOperator op) {
        assert elementwiseOperable(this);
        NDArray res = NDArray.zerosLike(this);
        float[] A = this.data;
        float[] B = res.data;

        int len = A.length;
        int i = 0;
        for (; i < SPECIES.loopBound(len); i += SPECIES_LEN) {
            var va = FloatVector.fromArray(SPECIES, A, i);
            va.lanewise(op.vectorOperator).intoArray(B, i);
        }

        for (; i < len; i++) {
            B[i] = op.processSingle(A[i]);
        }

        return res;
    }

    public NDArray performElementWiseOperator(NDArray other, ElementWiseBinaryOperator op) {
        NDArray res = NDArray.zerosLike(this);

        float[] A = this.data;
        float[] B = other.data;
        float[] C = res.data;

        int i = 0;
        for (; i < SPECIES.loopBound(data.length); i += SPECIES_LEN) {
            var va = FloatVector.fromArray(SPECIES, this.data, i);
            var vb = FloatVector.fromArray(SPECIES, other.data, i);
            va.lanewise(op.vectorOperator, vb).intoArray(C, i);
        }

        for (; i < data.length; i++) {
            C[i] = op.processSingle(A[i], B[i]);
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
            throw new IllegalArgumentException(
                    "can not convert to new shape, size: %d, newShape: %s"
                            .formatted(size, Arrays.toString(newShape)));
        }

        if (negIdx != -1) {
            newShape = Arrays.copyOf(newShape, newShape.length);
            newShape[negIdx] = size / prod;
        }

        if (getContiguous() == Flags.Contiguous.C) {
            return NDArray.of(newShape, data, Flags.Contiguous.C);
        }

        return NDArray.of(newShape, iterateToFloatArray(this, v -> v));
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
        String a = info() + "\n";
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
        return "NDArray Shape: %s Strides: %s Contiguous: %s"
                .formatted(Arrays.toString(shape), Arrays.toString(strides), getContiguous());
    }

    // test code
    public NDArray addNew(NDArray other) {
        assertShapesEqual(this, other);
        final ElementWiseBinaryOperator op = ElementWiseBinaryOperator.ADD;

        if (elementwiseOperable(this, other)) {
            return performElementWiseOperator(other, op);
        }
        return NDArray.of(shape, iterateToFloatArray(this, other, op::processSingle));
    }
}
