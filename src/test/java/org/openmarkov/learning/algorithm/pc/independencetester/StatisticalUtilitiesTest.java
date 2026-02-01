package org.openmarkov.learning.algorithm.pc.independencetester;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for StatisticalUtilities class using ground truth values.
 */
public class StatisticalUtilitiesTest {

    private static final double DELTA = 1e-6;

    @Test
    public void testGammaLn() {
        // ln(gamma(1)) = ln(0!) = ln(1) = 0
        assertEquals(0.0, StatisticalUtilities.gammaLn(1.0), DELTA, "gammaLn(1) should be 0");

        // ln(gamma(2)) = ln(1!) = 0
        assertEquals(0.0, StatisticalUtilities.gammaLn(2.0), DELTA, "gammaLn(2) should be 0");

        // ln(gamma(5)) = ln(4!) = ln(24) = 3.17805383
        assertEquals(3.17805383, StatisticalUtilities.gammaLn(5.0), DELTA, "gammaLn(5) should be approx 3.17805");

        // ln(gamma(0.5)) = ln(sqrt(pi)) = 0.5 * ln(pi) = 0.5723649429
        assertEquals(0.5723649429, StatisticalUtilities.gammaLn(0.5), DELTA, "gammaLn(0.5) failure");
    }

    @Test
    public void testChiSquareCDF() {
        // Critical values from Chi-Square distribution table (CDF values)

        // DF = 1
        // x = 3.841, p should be approx 0.95
        assertEquals(0.95, StatisticalUtilities.chiSquare(3.841, 1), 1e-3, "ChiSquare(3.841, 1) should be approx 0.95");

        // x = 2.706, p should be approx 0.90
        assertEquals(0.90, StatisticalUtilities.chiSquare(2.706, 1), 1e-3, "ChiSquare(2.706, 1) should be approx 0.90");

        // DF = 2
        // x = 5.991, p approx 0.95
        assertEquals(0.95, StatisticalUtilities.chiSquare(5.991, 2), 1e-3, "ChiSquare(5.991, 2) should be approx 0.95");

        // DF = 10
        // x = 18.307, p approx 0.95
        assertEquals(0.95, StatisticalUtilities.chiSquare(18.307, 10), 1e-3,
                "ChiSquare(18.307, 10) should be approx 0.95");
    }

    @Test
    public void testIncompleteGamma() {
        // gammp(a, x) is the regularized incomplete gamma function P(a,x)
        // Check consistency with ChiSquare: P(a, x) where a=df/2, x=stat/2

        // From WolframAlpha or R: pgamma(1, shape=1, scale=1) = 0.63212 (Exp(1)
        // distribution)
        // Here a=1, x=1
        assertEquals(0.63212056, StatisticalUtilities.gammp(1.0, 1.0), DELTA, "gammp(1,1)");

        // a=0.5 (related to chi-square df=1), x=0.5 (stat=1)
        // pgamma(0.5, shape=0.5, scale=1) = 0.682689
        assertEquals(0.68268949, StatisticalUtilities.gammp(0.5, 0.5), DELTA, "gammp(0.5, 0.5)");
    }

    @Test
    public void testEdgeCases() {
        assertThrows(IllegalArgumentException.class, () -> StatisticalUtilities.chiSquare(5.0, 0));
        assertThrows(IllegalArgumentException.class, () -> StatisticalUtilities.chiSquare(5.0, -1));

        assertEquals(0.0, StatisticalUtilities.chiSquare(0.0, 1), DELTA);
    }
}
