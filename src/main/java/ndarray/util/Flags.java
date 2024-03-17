package ndarray.util;

import java.util.Arrays;

public class Flags {

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
                    .orElseThrow(() -> new IllegalArgumentException("not a valid contiguous flag"));
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
