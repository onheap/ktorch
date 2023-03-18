package core.math

import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class MatrixTest {

    @Test
    fun testMk() {
        val a = mk.ndarray(
            mk[
                mk[1, 2, 3],
                mk[4, 5, 6],
            ]
        )
        val b = mk.ndarray(
            mk[
                mk[1, 2],
                mk[4, 5],
                mk[3, 6],
            ]
        )

        val c = a.dot(b)
        assertEquals(
            c,
            mk.ndarray(
                mk[
                    mk[18, 30],
                    mk[42, 69],
                ]
            )
        )
    }
}