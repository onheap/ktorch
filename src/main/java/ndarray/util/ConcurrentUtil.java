package ndarray.util;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

// https://github.com/lessthanoptimal/ejml/blob/SNAPSHOT/main/ejml-core/src/pabeles/concurrency/ConcurrencyOps.java
public class ConcurrentUtil {

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
