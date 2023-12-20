package playground;

import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;

public class Test {

    public static int[] simpleSum(int[] a, int[] b) {
        var c = new int[a.length];
        for (var i = 0; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }
        return c;
    }


    private static final VectorSpecies<Integer> SPECIES = IntVector.SPECIES_PREFERRED;
    public static int[] vectorSum(int[] a, int[] b) {
        var c = new int[a.length];
        var upperBound = SPECIES.loopBound(a.length);

        var i = 0;
        for (; i < upperBound; i += SPECIES.length()) {
            var va = IntVector.fromArray(SPECIES, a, i);
            var vb = IntVector.fromArray(SPECIES, b, i);
            var vc = va.add(vb);
            vc.intoArray(c, i);
        }
        // Compute elements not fitting in the vector alignment.
        for (; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }

        return c;
    }
}
