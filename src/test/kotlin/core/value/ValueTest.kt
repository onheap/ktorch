package core.value

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

internal class ValueTest {
    companion object {
        const val DELTA = 0.0001
    }

    @Test
    fun plus() {
        val a = Value(3.0)

        val b = Value(3.0)

        val x = a + 4.0
        val y = b + Value(4.0)

        assertEquals(7.0, x.data, DELTA)
        assertEquals(7.0, y.data, DELTA)
    }

    @Test
    fun times() {
        val a = Value(3.0)
        val b = Value(3.0)

        val x = a * 4.0
        val y = b * Value(4.0)

        x.backward()
        y.backward()

        assertEquals(12.0, x.data, DELTA)
        assertEquals(12.0, y.data, DELTA)

        assertEquals(4.0, a.grad, DELTA)
        assertEquals(4.0, b.grad, DELTA)
    }

    @Test
    fun power() {
        val a = Value(3.0)
        val b = Value(3.0)

        val x = a.pow(4)
        val y = b.pow(4.0)

        x.backward()
        y.backward()

        assertEquals(81.0, x.data, DELTA)
        assertEquals(81.0, y.data, DELTA)

        assertEquals(108.0, a.grad, DELTA)
        assertEquals(108.0, b.grad, DELTA)
    }

    @Test
    fun backward1() {
        val x = Value(-4.0)
        val z = 2 * x + 2 + x
        val q = z.relu() + z * x
        val h = (z * z).relu()
        val y = h + q + q * x
        y.backward()

        assertEquals(-4.0, x.data, DELTA)
        assertEquals(-20.0, y.data, DELTA)
        assertEquals(46.0, x.grad, DELTA)
    }

    @Test
    fun backward2() {
        var a = Value(-4.0)
        var b = Value(2.0)

        var c = a + b
        var d = a * b + b.pow(3)
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).relu()
        d = d + 3 * d + (b - a).relu()
        var e = c - d
        var f = e.pow(2)
        var g = f / 2.0
        g = g + 10.0 / f
        g.backward()

        assertEquals(24.7041, g.data, DELTA)

        assertEquals(138.8338, a.grad, DELTA)
        assertEquals(645.5773, b.grad, DELTA)
    }

    @Test
    fun sum() {
        val l1 =
            listOf(
                Value(2.0),
                Value(1.0),
                Value(-3.0),
                Value(45.0),
                Value(28.0),
                Value(82.0),
                Value(-5.0),
                Value(9.0),
                Value(-33.0),
                Value(51.0),
                Value(38.0),
                Value(78.0))

        val l2 =
            listOf(
                Value(2.0),
                Value(1.0),
                Value(-3.0),
                Value(45.0),
                Value(28.0),
                Value(82.0),
                Value(-5.0),
                Value(9.0),
                Value(-33.0),
                Value(51.0),
                Value(38.0),
                Value(78.0))

        val res1 =
            l1[0] +
                l1[1] +
                l1[2] +
                l1[3] +
                l1[4] +
                l1[5] +
                l1[6] +
                l1[7] +
                l1[8] +
                l1[9] +
                l1[10] +
                l1[11]

        val res2 = l2.sum()

        res1.backward()
        res2.backward()

        assertEquals(res1.data, res2.data, DELTA)
        assertEquals(res1.grad, res2.grad, DELTA)

        for (i in 0..11) {
            val v1 = l1[i]
            val v2 = l2[i]

            assertEquals(v1.data, v2.data, DELTA)
            assertEquals(v1.grad, v2.grad, DELTA)
        }
    }
}
