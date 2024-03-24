package ndarray;

import static ndarray.util.ShapeUtil.copyIndices;
import static ndarray.util.ShapeUtil.reduceShape;
import static ndarray.util.Util.reverseArray;

import java.util.Iterator;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import ndarray.iterator.IndicesIterator;
import ndarray.iterator.NDArrayIterator;
import ndarray.operator.ElementWiseBinaryOperator;
import ndarray.operator.ElementWiseReduceOperator;
import ndarray.operator.ElementWiseUnaryOperator;
import ndarray.util.Flags;
import ndarray.util.PresentUtil;
import ndarray.util.ShapeUtil;

public class NDArray implements Iterable<Float> {

    public record Data(float[] array, int offset) {}

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int SPECIES_LEN = SPECIES.length();

    final float[] data;
    final int offset;

    final int[] shape;
    final int[] strides;
    final byte flags;

    public NDArray(int[] shape, float[] data) {
        this.shape = shape;
        this.data = data;
        this.offset = 0;
        this.strides = Flags.Contiguous.C.calculateStrides(shape);
        this.flags = Flags.setContiguous(Flags.ZERO, Flags.Contiguous.C);
    }

    public NDArray(int[] shape, int[] strides, float[] data, int offset, byte flags) {
        this.shape = shape;
        this.data = data;
        this.offset = offset;
        this.strides = strides;
        this.flags = flags;
    }

    public Data getData() {
        return new Data(data, offset);
    }

    public int[] getShape() {
        return shape;
    }

    public int[] getStrides() {
        return strides;
    }

    public int getSize() {
        return ShapeUtil.getSize(shape);
    }

    public int getDim() {
        return shape.length;
    }

    public Iterator<Float> iterator() {
        return new NDArrayIterator(this);
    }

    public Iterable<int[]> indices() {
        return new IndicesIterator(this).iterable();
    }

    public Flags.Contiguous getContiguous() {
        return Flags.getContiguous(flags);
    }

    public boolean isScalar() {
        return shape.length == 0 && getSize() == 1;
    }

    public float asScalar() {
        if (!isScalar()) {
            throw new IllegalArgumentException("Not a scalar");
        }
        return data[offset];
    }

    public NDArray transpose() {
        int[] transposedShape = reverseArray(this.shape);
        int[] transposedStrides = reverseArray(this.strides);
        return new NDArray(
                transposedShape,
                transposedStrides,
                data,
                offset,
                Flags.setContiguous(Flags.ZERO, transposedShape, transposedStrides));
    }

    public NDArray matmul(NDArray other) {
        return MatMul.matmul(this, other);
    }

    public NDArray sum() {
        float total = ElementWiseReduceOperator.SUM.elementWiseReduce(data, offset, data.length);
        return NDArrays.ofScalar(total);
    }

    public NDArray sum(int dim) {
        return ElementWiseReduceOperator.SUM.reduceAlongDimension(this, dim, false);
    }

    public NDArray sum(int dim, boolean keepDims) {
        return ElementWiseReduceOperator.SUM.reduceAlongDimension(this, dim, keepDims);
    }

    public NDArray max() {
        float max = ElementWiseReduceOperator.MAX.elementWiseReduce(data, offset, data.length);
        return NDArrays.ofScalar(max);
    }

    public NDArray max(int dim) {
        return ElementWiseReduceOperator.MAX.reduceAlongDimension(this, dim, false);
    }

    public NDArray max(int dim, boolean keepDims) {
        return ElementWiseReduceOperator.MAX.reduceAlongDimension(this, dim, keepDims);
    }

    public NDArray mean() {
        return NDArrays.ofScalar(sum().asScalar() / getSize());
    }

    public NDArray argmax() {
        if (getContiguous() == Flags.Contiguous.C) {
            int maxIndex = offset;
            float max = data[offset];
            for (int i = offset; i < offset + getSize(); i++) {
                if (data[i] > max) {
                    maxIndex = i;
                    max = data[i];
                }
            }
            return NDArrays.ofScalar(maxIndex - offset);
        }

        int len = shape.length;
        int[] maxIndices = new int[len];
        float max = data[offset];

        for (var indices : indices()) {
            float v = get(indices);
            if (v > max) {
                max = v;
                System.arraycopy(indices, 0, maxIndices, 0, len);
            }
        }

        int base = 1;
        int res = maxIndices[len - 1];
        for (int i = len - 2; i >= 0; i--) {
            base = base * shape[i + 1];
            res += maxIndices[i] * base;
        }

        return NDArrays.ofScalar(res);
    }

    public NDArray argmax(int dim) {
        return argmax(dim, false);
    }

    public NDArray argmax(int dim, boolean keepDims) {
        int len = shape.length;
        dim = dim < 0 ? len + dim : dim;

        if (dim < 0 || dim >= len) {
            throw new IllegalArgumentException(
                    "dim %d is out of bounds for array of dimension %d".formatted(dim, len));
        }

        int[] newShape = reduceShape(shape, dim, keepDims);

        NDArray res = NDArrays.fill(newShape, 0);
        int[] dimMaxIndices = new int[len];
        int[] resIndices = new int[res.shape.length];
        for (int[] indices : this.indices()) {

            copyIndices(indices, resIndices, dim, keepDims);

            float v = this.get(indices);
            int dimMaxIdx = (int) res.get(resIndices);
            System.arraycopy(indices, 0, dimMaxIndices, 0, len);
            dimMaxIndices[dim] = dimMaxIdx;
            if (v > get(dimMaxIndices)) {
                res.set(resIndices, indices[dim]);
            }
        }
        return res;
    }

    public NDArray maximum(float v) {
        return this.maximum(NDArrays.ofScalar(v));
    }

    public NDArray maximum(NDArray other) {
        return ElementWiseBinaryOperator.MAX.performBinaryOperator(this, other);
    }

    public NDArray minimum(float v) {
        return this.minimum(NDArrays.ofScalar(v));
    }

    public NDArray minimum(NDArray other) {
        return ElementWiseBinaryOperator.MIN.performBinaryOperator(this, other);
    }

    public NDArray add(float other) {
        return this.add(NDArrays.ofScalar(other));
    }

    public NDArray add(NDArray other) {
        return ElementWiseBinaryOperator.ADD.performBinaryOperator(this, other);
    }

    public NDArray sub(float other) {
        return this.sub(NDArrays.ofScalar(other));
    }

    public NDArray sub(NDArray other) {
        return ElementWiseBinaryOperator.SUB.performBinaryOperator(this, other);
    }

    public NDArray mul(float other) {
        return this.mul(NDArrays.ofScalar(other));
    }

    public NDArray mul(NDArray other) {
        return ElementWiseBinaryOperator.MUL.performBinaryOperator(this, other);
    }

    public NDArray div(float other) {
        return this.div(NDArrays.ofScalar(other));
    }

    public NDArray div(NDArray other) {
        return ElementWiseBinaryOperator.DIV.performBinaryOperator(this, other);
    }

    public NDArray log() {
        return ElementWiseUnaryOperator.LOG.performElementwise(this);
    }

    public NDArray exp() {
        return ElementWiseUnaryOperator.EXP.performElementwise(this);
    }

    public NDArray reshape(int... newShape) {
        newShape = ShapeUtil.reshape(shape, newShape);

        if (getContiguous() == Flags.Contiguous.C) {
            return NDArrays.of(newShape, data, offset, Flags.Contiguous.C);
        }

        // can we make it non-copying?
        return NDArrays.of(newShape, NDArrays.performIteratively(this, v -> v));
    }

    private int getFlatIndex(int[] indices) {
        return ShapeUtil.getFlatIndex(indices, strides);
    }

    private int[] getIndices(int flatIndex) {
        return ShapeUtil.getIndices(flatIndex, shape);
    }

    public float get(int[] indices) {
        return data[offset + getFlatIndex(indices)];
    }

    public void set(int[] indices, float v) {
        data[offset + getFlatIndex(indices)] = v;
    }

    public NDArray getNDArray(int[] indices) {
        int[] subShape = ShapeUtil.getSubShape(shape, indices);
        int[] subStrides = ShapeUtil.getSubStrides(strides, indices);
        int subOffset = offset + getFlatIndex(indices);

        return new NDArray(
                subShape,
                subStrides,
                data,
                subOffset,
                Flags.setContiguous(
                        Flags.ZERO,
                        getContiguous() == Flags.Contiguous.C
                                ? Flags.Contiguous.C
                                : Flags.Contiguous.NOT // Not Contiguous?
                        ));
    }

    public float[] toArray() {
        return PresentUtil.toArray(this);
    }

    public float[][] toMatrix() {
        return PresentUtil.toMatrix(this);
    }

    @Override
    public String toString() {
        return PresentUtil.toString(this);
    }

    public String info() {
        return PresentUtil.info(this);
    }

    // performance test code
    public NDArray addIterative(NDArray other) {
        return NDArrays.performIteratively(this, other, (a, b) -> a + b);
    }

    public NDArray addBroadcast(NDArray other) {
        return NDArrays.performBroadcastly(this, other, (a, b) -> a + b);
    }

    public NDArray addVector(NDArray other) {
        NDArray res = NDArrays.zerosLike(this);

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
}
