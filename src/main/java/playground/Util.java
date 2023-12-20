package playground;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

public class Util {

    private Util() {
    }

    public static int[] arrOf(int... a) {
        return a;
    }

    public static float[] arrOf(float... a) {
        return a;
    }

    public static int[] arrOfI(int... a) {
        return a;
    }

    public static float[] arrOfF(float... a) {
        return a;
    }

    // https://github.com/lessthanoptimal/ejml/blob/SNAPSHOT/main/ejml-core/src/pabeles/concurrency/ConcurrencyOps.java
    private static final ForkJoinPool POOL = new ForkJoinPool();

    public static void concurrentLoopFor(int start, int endExclusive, IntConsumer consumer) {
        try {
            POOL.submit(() -> IntStream.range(start, endExclusive).parallel().forEach(consumer)).get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }
}
