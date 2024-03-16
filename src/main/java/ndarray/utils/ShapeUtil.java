package ndarray.utils;

import static ndarray.utils.Util.arrOf;

import java.util.Arrays;

// https://github.com/mikex86/scicore/blob/master/core/src/main/java/me/mikex86/scicore/utils/ShapeUtils.java
public class ShapeUtil {

    public static int getSize(int[] shape) {
        return Arrays.stream(shape).reduce(1, (a, b) -> a * b);
    }

    public static int getFlatIndex(int[] indices, int[] strides) {
        int index = 0;
        for (int i = 0; i < indices.length; i++) {
            index += indices[i] * strides[i];
        }
        return index;
    }

    public static int[] getIndices(int flatIndex, int[] shape) {
        return calculateIndices(flatIndex, shape, new int[shape.length]);
    }

    public static int[] calculateIndices(int flatIndex, int[] shape, int[] indices) {
        for (int j = shape.length - 1; j >= 0; j--) {
            indices[j] = flatIndex % shape[j];
            flatIndex /= shape[j];
        }
        return indices;
    }

    public static boolean increaseIndices(int[] indices, int[] shape) {
        for (int i = indices.length - 1; i >= 0; i--) {
            if (indices[i] < shape[i] - 1) {
                indices[i]++;
                return true;
            } else {
                indices[i] = 0;
            }
        }

        return false;
    }

    public static void copyIndices(int[] indices, int[] resIndices, int dim, boolean keepDims) {
        int len = indices.length;
        if (0 < dim) {
            System.arraycopy(indices, 0, resIndices, 0, dim);
        }

        if ((dim + 1) <= (len - 1)) {
            System.arraycopy(
                    indices,
                    dim + 1,
                    resIndices,
                    keepDims ? dim + 1 : dim,
                    (len - 1) - (dim + 1) + 1);
        }
    }

    public static int[] constrainIndices(int[] indices, int[] shape) {
        for (int i = 0; i < indices.length; i++) {
            indices[i] %= shape[i];
        }
        return indices;
    }

    public static int[] reduceShape(int[] shape, int dim, boolean keepDims) {
        if (keepDims) {
            int[] newShape = Arrays.copyOf(shape, shape.length);
            newShape[dim] = 1;
            return newShape;
        }
        return reduceShape(shape, dim);
    }

    public static int[] reduceShape(int[] shape, int dim) {
        int len = shape.length;
        int[] newShape = new int[len - 1];

        if (0 < dim) {
            System.arraycopy(shape, 0, newShape, 0, dim);
        }

        if ((dim + 1) <= (len - 1)) {
            System.arraycopy(shape, dim + 1, newShape, dim, (len - 1) - (dim + 1) + 1);
        }

        return newShape;
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

    public static boolean shapesEqual(int[] shapeA, int[] shapeB) {
        return Arrays.equals(shapeA, shapeB);
    }

    public static boolean isBroadcastedShapes(int[] shapeA, int[] shapeB) {
        if (shapesEqual(shapeA, shapeB)) {
            return false;
        }

        if (shapeA.length < shapeB.length) {
            int[] temp = shapeA;
            shapeA = shapeB;
            shapeB = temp;
        }

        int diff = shapeA.length - shapeB.length;
        int len = shapeA.length;

        for (int i = len - 1; i >= 0; i--) {
            int j = i - diff;
            if (j >= 0 && (shapeA[i] != shapeB[j] && shapeA[i] != 1 && shapeB[j] != 1)) {
                return false;
            }
        }
        return true;
    }

    public static void main(String[] args) {
        var res = isBroadcastedShapes(arrOf(1, 2, 3), arrOf(2, 3));
        System.out.println(res);
    }
}
