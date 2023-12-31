package ndarray;

import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

public class Util {

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
                            .formatted(Arrays.toString(a.shape), Arrays.toString(b.shape)));
        }
    }

    public static boolean shapesEqual(NDArray a, NDArray b) {
        return ShapeUtil.shapesEqual(a.shape, b.shape);
    }

    public static boolean elementwiseOperable(NDArray a, NDArray b) {
        if (Flags.getContiguous(a.flags) == Flags.Contiguous.NOT
                || Flags.getContiguous(b.flags) == Flags.Contiguous.NOT) {
            return false;
        }

        if ((Flags.isCContiguous(a.flags) && !Flags.isCContiguous(b.flags))
                || (Flags.isFContiguous(a.flags) && !Flags.isFContiguous(b.flags))) {
            return false;
        }

        if (!Arrays.equals(a.shape, b.shape)) {
            return false;
        }

        if (!Arrays.equals(a.strides, b.strides)) {
            return false;
        }

        return true;
    }

    public static boolean elementwiseOperable(NDArray a) {
        return Flags.getContiguous(a.flags) != Flags.Contiguous.NOT;
    }

    public static class Flags {

        public static final byte ZERO = (byte) 0;

        public enum Contiguous {
            MASK((byte) 0b11) {
                @Override
                public int[] calculateStrides(int[] shape) {
                    throw new UnsupportedOperationException();
                }
            },

            // Not Contiguous
            NOT((byte) 0b00) {
                @Override
                public int[] calculateStrides(int[] shape) {
                    throw new UnsupportedOperationException();
                }
            },

            C((byte) 0b01) {
                @Override
                public int[] calculateStrides(int[] shape) {
                    int[] strides = new int[shape.length];
                    int stride = 1;
                    for (int i = shape.length - 1; i >= 0; i--) {
                        strides[i] = stride;
                        stride *= shape[i];
                    }
                    return strides;
                }
            },
            F((byte) 0b10) {
                @Override
                public int[] calculateStrides(int[] shape) {
                    int[] strides = new int[shape.length];
                    int stride = 1;
                    for (int i = 0; i < shape.length; i++) {
                        strides[i] = stride;
                        stride *= shape[i];
                    }
                    return strides;
                }
            };

            private final byte flag;

            public abstract int[] calculateStrides(int[] shape);

            public static Contiguous of(byte flag) {
                return Arrays.stream(Contiguous.values())
                        .filter(con -> con.flag == flag)
                        .findFirst()
                        .orElseThrow(
                                () -> new IllegalArgumentException("not a valid contiguous flag"));
            }

            Contiguous(byte flag) {
                this.flag = flag;
            }
        }

        public static byte setContiguous(byte flags, int[] shape, int[] strides) {
            if (isValidCContiguous(shape, strides)) {
                flags = setContiguous(flags, Contiguous.C);
            } else if (isValidFContiguous(shape, strides)) {
                flags = setContiguous(flags, Contiguous.F);
            } else {
                // flags = setContiguous(flags, Contiguous.NOT);
                throw new IllegalArgumentException("Not a valid contiguous");
            }

            return flags;
        }

        public static byte setContiguous(byte flags, Contiguous contiguous) {
            flags &= ~Contiguous.MASK.flag;
            flags |= contiguous.flag;
            return flags;
        }

        public static Contiguous getContiguous(byte flags) {
            return Contiguous.of((byte) (flags & Contiguous.MASK.flag));
        }

        public static boolean isCContiguous(byte flags) {
            return (flags & Contiguous.MASK.flag) == Contiguous.C.flag;
        }

        public static boolean isFContiguous(byte flags) {
            return (flags & Contiguous.MASK.flag) == Contiguous.F.flag;
        }

        public static boolean isValidCContiguous(int[] shape, int[] strides) {
            for (int i = 0; i < shape.length - 1; i++) {
                if (strides[i] != strides[i + 1] * shape[i + 1]) {
                    return false;
                }
            }
            return true;
        }

        public static boolean isValidFContiguous(int[] shape, int[] strides) {
            for (int i = 1; i < shape.length; i++) {
                if (strides[i] != strides[i - 1] * shape[i - 1]) {
                    return false;
                }
            }
            return true;
        }
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
