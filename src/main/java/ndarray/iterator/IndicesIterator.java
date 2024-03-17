package ndarray.iterator;

import static ndarray.util.ShapeUtil.calculateIndices;

import java.util.Iterator;
import java.util.NoSuchElementException;
import ndarray.NDArray;

public class IndicesIterator implements Iterator<int[]> {
    private int curt = 0;
    private final int size;

    private final int[] shape;

    private final int[] indices;

    public IndicesIterator(NDArray ndArray) {
        this.size = ndArray.getSize();
        this.shape = ndArray.getShape();
        this.indices = new int[shape.length];
    }

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

        return calculateIndices(curt++, shape, indices);
    }

    public Iterable<int[]> iterable() {
        return () -> this;
    }
}
