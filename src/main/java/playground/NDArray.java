package playground;

import java.text.DecimalFormat;

public class NDArray {
    float[] data;
    int[] shape;
    int[] strides;

    public NDArray(float[] data, int[] shape) {
        this.data = data;
        this.shape = shape;
        this.strides = calculateStrides(shape);
    }

    // 计算步长
    private int[] calculateStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }


    public NDArray broadcast(NDArray other) {
        return null;
    }

    // 实现点积运算
    public NDArray dot(NDArray other) {
        return null;
    }


    private int getFlatIndex(int[] indices) {
        int index = 0;
        for (int i = 0; i < indices.length; i++) {
            index += indices[i] * strides[i];
        }
        return index;
    }

    @Override
    public String toString() {

        var df = new DecimalFormat("##.####");

        String a = "";
        a += "[".repeat(shape.length);
        int indent = shape.length;

        for (int i = 0; i < data.length; i++) {
            float f = data[i];
            a += df.format(f);
        }

      return a;
    }

    public static void main(String[] args) {
        float[] data = new float[]{1, 2, 3, 4, 5, 6};
        int[] shape = new int[]{2, 3};
        NDArray a = new NDArray(data, shape);
        System.out.println(a);
    }
}