package ndarray.utils;

import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import ndarray.Flags;
import ndarray.NDArray;

public class Util {

    public static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    public static final int SPECIES_LEN = SPECIES.length();

    @FunctionalInterface
    public interface FloatUnaryOperator {
        float applyAsFloat(float operand);
    }

    @FunctionalInterface
    public interface FloatBinaryOperator {
        float applyAsFloat(float a, float b);
    }

    public static int[] arrOf(int... a) {
        return a;
    }

    public static float[] arrOf(float... a) {
        return a;
    }

    public static float[] arrOfF(int... a) {
        float[] res = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            res[i] = a[i];
        }
        return res;
    }

    public static int[] reverseArray(int[] A) {
        int[] array = Arrays.copyOf(A, A.length);
        for (int i = 0; i < array.length / 2; i++) {
            int temp = array[i];
            array[i] = array[array.length - 1 - i];
            array[array.length - 1 - i] = temp;
        }
        return array;
    }

    public static void assertShapesEqual(NDArray a, NDArray b) {
        if (!shapesEqual(a, b)) {
            throw new IllegalArgumentException(
                    "shapes not equal, a: %s, b: %s"
                            .formatted(
                                    Arrays.toString(a.getShape()), Arrays.toString(b.getShape())));
        }
    }

    public static boolean shapesEqual(NDArray a, NDArray b) {
        return ShapeUtil.shapesEqual(a.getShape(), b.getShape());
    }

    public static boolean elementwiseOperable(NDArray a, NDArray b) {
        if (a.getContiguous() == Flags.Contiguous.NOT
                || b.getContiguous() == Flags.Contiguous.NOT) {
            return false;
        }

        if ((a.getContiguous() != b.getContiguous())) {
            return false;
        }

        if (!Arrays.equals(a.getShape(), b.getShape())) {
            return false;
        }

        if (!Arrays.equals(a.getStrides(), b.getStrides())) {
            return false;
        }

        return true;
    }

    public static boolean elementwiseOperable(NDArray a) {
        return a.getContiguous() != Flags.Contiguous.NOT;
    }

    public static class Concurrent {
        // https://github.com/lessthanoptimal/ejml/blob/SNAPSHOT/main/ejml-core/src/pabeles/concurrency/ConcurrencyOps.java
        private static final ForkJoinPool POOL = new ForkJoinPool();

        public static void loopFor(int start, int endExclusive, IntConsumer consumer) {
            try {
                POOL.submit(() -> IntStream.range(start, endExclusive).parallel().forEach(consumer))
                        .get();
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
        }
    }
}
