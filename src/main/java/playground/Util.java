package playground;

import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

public class Util {
    public static int[] arrOf(int... a) {
        return a;
    }

    public static float[] arrOf(float... a) {
        return a;
    }

    public static float[] arrOfF(float... a) {
        return a;
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

    public static class Flags {

        private static final byte FLAG_ORDER_MASK = 0b11;
        private static final byte FLAG_C_ORDER = 0b01;
        private static final byte FLAG_F_ORDER = 0b10;
        private static final byte FLAG_S_ORDER = 0b11;


        public static byte setOrder(byte flags, int[] shape, int[] strides) {
            if (isValidCOrder(shape, strides)) {
                flags = setCOrder(flags);
            } else if (isValidFOrder(shape, strides)) {
                flags = setFOrder(flags);
            }

            return flags;
        }

        public static byte setCOrder(byte flags) {
            flags &= ~FLAG_ORDER_MASK;
            flags |= FLAG_C_ORDER;
            return flags;
        }

        public static byte setFOrder(byte flags) {
            flags &= ~FLAG_ORDER_MASK;
            flags |= FLAG_F_ORDER;
            return flags;
        }

        public static byte setSOrder(byte flags) {
            flags &= ~FLAG_ORDER_MASK;
            flags |= FLAG_S_ORDER;
            return flags;
        }

        public static boolean isCOrder(byte flags) {
            return (flags & FLAG_ORDER_MASK) == FLAG_C_ORDER;
        }

        public static boolean isFOrder(byte flags) {
            return (flags & FLAG_ORDER_MASK) == FLAG_F_ORDER;
        }

        public static boolean isValidCOrder(int[] shape, int[] strides) {
            for (int i = 0; i < shape.length - 1; i++) {
                if (strides[i] != strides[i + 1] * shape[i + 1]) {
                    return false;
                }
            }
            return true;
        }

        public static boolean isValidFOrder(int[] shape, int[] strides) {
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
                POOL.submit(() -> IntStream.range(start, endExclusive).parallel().forEach(consumer)).get();
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
        }
    }
}
