package ndarray.iterator;

import static ndarray.util.ShapeUtil.calculateIndices;

import java.util.Iterator;
import java.util.NoSuchElementException;
import ndarray.NDArray;
import ndarray.util.Flags;

public class NDArrayIterator implements Iterator<Float> {
    private int curt = 0;

    private final NDArray ndArray;

    private final int size;
    private final boolean isCContiguous;

    private final int[] shape;
    private final int[] indices;

    private final float[] data;

    public NDArrayIterator(NDArray ndArray) {
        this.ndArray = ndArray;
        this.size = ndArray.getSize();
        this.isCContiguous = ndArray.getContiguous() == Flags.Contiguous.C;
        this.shape = ndArray.getShape();
        this.indices = isCContiguous ? null : new int[shape.length];
        this.data = ndArray.getData();
    }

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
            return ndArray.get(calculateIndices(curt++, shape, indices));
        }
    }

    public Iterable<Float> iterable() {
        return () -> this;
    }
}
