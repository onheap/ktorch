package ndarray.util;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Iterator;
import java.util.function.Function;
import java.util.stream.Collectors;
import ndarray.NDArray;

public class PresentUtil {

    public static float[] toArray(NDArray a) {
        if (a.getContiguous() == Flags.Contiguous.C) {
            return Arrays.copyOf(a.getData(), a.getData().length);
        }

        Iterator<Float> it = a.iterator();
        float[] res = new float[a.getSize()];
        for (int i = 0; i < res.length; i++) {
            res[i] = it.next();
        }
        return res;
    }

    public static float[][] toMatrix(NDArray a) {
        int[] shape = a.getShape();
        if (shape.length != 2) {
            throw new IllegalStateException(
                    "Unable to convert to Matrix: shape: %s".formatted(Arrays.toString(shape)));
        }

        int m = shape[0];
        int n = shape[1];
        float[][] res = new float[m][n];

        if (a.getContiguous() == Flags.Contiguous.C) {
            for (int i = 0; i < m; i++) {
                System.arraycopy(a.getData(), i * n, res[i], 0, n);
            }
            return res;
        }

        for (int[] idx : a.indices()) {
            res[idx[0]][idx[1]] = a.get(idx);
        }

        return res;
    }

    public static String toString(NDArray a) {
        final var df = new DecimalFormat("##.####");
        // String s = info() + "\n";
        StringBuilder s = new StringBuilder();
        int len = a.dim();

        s.append("[".repeat(len));

        boolean first = true;

        for (int[] indices : a.indices()) {
            if (!first) {
                int zeros = 0;
                for (int j = len - 1; j >= 0; j--) {
                    if (indices[j] == 0) {
                        zeros++;
                    } else {
                        break;
                    }
                }

                if (zeros > 0) {
                    s.append("]".repeat(zeros));
                    s.append("\n");
                    s.append(" ".repeat(len - zeros));
                    s.append("[".repeat(zeros));
                } else {
                    s.append(", ");
                }
            }
            first = false;
            var f = a.get(indices);
            s.append(df.format(f));
        }

        s.append("]".repeat(len));

        return s.toString();
    }

    public static String info(NDArray a) {
        final Function<int[], String> joinToStr =
                (int[] arr) ->
                        Arrays.stream(a.getShape())
                                .mapToObj(String::valueOf)
                                .collect(Collectors.joining(",", "(", ")"));

        return "NDArray: %s %s %s"
                .formatted(
                        joinToStr.apply(a.getShape()),
                        joinToStr.apply(a.getStrides()),
                        a.getContiguous());
    }
}
