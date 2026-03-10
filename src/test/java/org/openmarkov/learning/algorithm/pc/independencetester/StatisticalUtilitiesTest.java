package org.openmarkov.learning.algorithm.pc.independencetester;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Test suite for {@link StatisticalUtilities}.
 *
 * <p>This class validates the numerical and semantic correctness of the
 * statistical helper methods used by the PC independence testing code.
 * In particular, it verifies the following groups of properties:
 *
 * <ul>
 *   <li><b>Boundary conditions</b>, such as the values of the regularized
 *       incomplete gamma functions at {@code x = 0}.</li>
 *   <li><b>Fundamental mathematical identities</b>, such as
 *       {@code P(a,x) + Q(a,x) = 1} and the analogous relationship between
 *       the chi-square cumulative distribution function and its right-tail
 *       probability.</li>
 *   <li><b>Closed-form special cases</b>, including the exponential form
 *       obtained when {@code a = 1} and the chi-square distribution with
 *       two degrees of freedom.</li>
 *   <li><b>Regression checks</b> specifically designed to detect accidental
 *       confusion between chi-square CDF values and right-tail p-values.</li>
 *   <li><b>Known values of {@code ln(Gamma(x))}</b> for arguments with
 *       exact factorial interpretations.</li>
 *   <li><b>Input validation</b>, ensuring that invalid arguments are
 *       rejected consistently with {@link IllegalArgumentException}.</li>
 *   <li><b>Range preservation</b>, ensuring that probability-like quantities
 *       always remain within the interval {@code [0,1]}.</li>
 * </ul>
 *
 * <p>The tests use two tolerances:
 *
 * <ul>
 *   <li>A stricter tolerance for exact or near-exact identities.</li>
 *   <li>A looser tolerance for numerical approximations based on series or
 *       continued-fraction computations.</li>
 * </ul>
 *
 * <p>These tests are intentionally redundant in some areas. That redundancy is
 * useful here because the code under test implements mathematically delicate
 * numerical routines. Several tests target similar functionality from different
 * angles in order to detect:
 *
 * <ul>
 *   <li>incorrect formulas,</li>
 *   <li>loss of normalization,</li>
 *   <li>incorrect tail interpretation,</li>
 *   <li>regressions caused by refactoring, and</li>
 *   <li>numerical drift outside valid probability bounds.</li>
 * </ul>
 *
 *  @author Manuel Arias
 */
class StatisticalUtilitiesTest {

    /**
     * Strict tolerance used for assertions that are expected to be exact
     * in theory and very stable numerically in practice.
     *
     * <p>This value is used for tests such as:
     * <ul>
     *   <li>{@code P(a,0) = 0},</li>
     *   <li>{@code Q(a,0) = 1},</li>
     *   <li>chi-square CDF at zero,</li>
     *   <li>chi-square p-value at zero.</li>
     * </ul>
     */
    private static final double TOLERANCE = 1e-10;

    /**
     * Looser tolerance used for comparisons involving iterative numerical
     * approximations, such as the regularized incomplete gamma function,
     * the chi-square distribution, and the logarithm of the gamma function.
     *
     * <p>The methods under test rely on approximations whose floating-point
     * behaviour is correct but not exact to machine precision in all cases.
     * This tolerance gives enough margin for stable numerical validation
     * without making the tests too weak.
     */
    private static final double LOOSE_TOLERANCE = 1e-7;

    /**
     * Verifies the lower regularized incomplete gamma function at the boundary
     * point {@code x = 0}.
     *
     * <p>Mathematically,
     *
     * <pre>
     * P(a,0) = 0    for every a > 0
     * </pre>
     *
     * because the lower incomplete gamma integral over the interval
     * {@code [0,0]} is zero.
     *
     * <p>This test checks several representative values of {@code a} in order
     * to ensure that the implementation handles the zero boundary correctly
     * regardless of the shape parameter.
     */
    @Test
    @DisplayName("regularizedGammaP(a, 0) = 0")
    void regularizedGammaPAtZeroShouldBeZero() {
        assertEquals(0.0, StatisticalUtilities.regularizedGammaP(2.0, 0.0), TOLERANCE);
        assertEquals(0.0, StatisticalUtilities.regularizedGammaP(0.5, 0.0), TOLERANCE);
        assertEquals(0.0, StatisticalUtilities.regularizedGammaP(10.0, 0.0), TOLERANCE);
    }

    /**
     * Verifies the upper regularized incomplete gamma function at the boundary
     * point {@code x = 0}.
     *
     * <p>Mathematically,
     *
     * <pre>
     * Q(a,0) = 1    for every a > 0
     * </pre>
     *
     * because the entire normalized mass lies in the upper tail when the
     * lower integration limit is zero.
     *
     * <p>This test complements the previous one and confirms that both
     * regularized gamma functions are initialized consistently at the left
     * boundary of their domain.
     */
    @Test
    @DisplayName("regularizedGammaQ(a, 0) = 1")
    void regularizedGammaQAtZeroShouldBeOne() {
        assertEquals(1.0, StatisticalUtilities.regularizedGammaQ(2.0, 0.0), TOLERANCE);
        assertEquals(1.0, StatisticalUtilities.regularizedGammaQ(0.5, 0.0), TOLERANCE);
        assertEquals(1.0, StatisticalUtilities.regularizedGammaQ(10.0, 0.0), TOLERANCE);
    }

    /**
     * Verifies the complementarity identity of the regularized incomplete
     * gamma functions:
     *
     * <pre>
     * P(a,x) + Q(a,x) = 1
     * </pre>
     *
     * <p>This is one of the most important structural identities of the gamma
     * functions implemented in {@link StatisticalUtilities}. If this identity
     * fails, then one of the following is likely wrong:
     *
     * <ul>
     *   <li>the series approximation,</li>
     *   <li>the continued-fraction approximation,</li>
     *   <li>the switching criterion between both approximations, or</li>
     *   <li>the normalization by {@code Gamma(a)}.</li>
     * </ul>
     *
     * <p>The test uses representative values spanning:
     * <ul>
     *   <li>small and moderate values of {@code a},</li>
     *   <li>small, medium and large values of {@code x},</li>
     *   <li>regions where different internal numerical methods are expected
     *       to be used.</li>
     * </ul>
     */
    @Test
    @DisplayName("P(a,x) + Q(a,x) = 1")
    void regularizedGammaPAndQShouldAddUpToOne() {
        double[][] cases = {
                {0.5, 0.1},
                {0.5, 1.0},
                {0.5, 5.0},
                {1.0, 0.1},
                {1.0, 1.0},
                {2.0, 1.0},
                {2.0, 5.0},
                {5.0, 2.0},
                {5.0, 10.0},
                {10.0, 12.0}
        };

        for (double[] testCase : cases) {
            double a = testCase[0];
            double x = testCase[1];
            double p = StatisticalUtilities.regularizedGammaP(a, x);
            double q = StatisticalUtilities.regularizedGammaQ(a, x);
            assertEquals(1.0, p + q, LOOSE_TOLERANCE,
                    "Failed for a=" + a + ", x=" + x);
        }
    }

    /**
     * Verifies a closed-form special case of the lower regularized incomplete
     * gamma function.
     *
     * <p>For {@code a = 1}, the regularized lower incomplete gamma function
     * simplifies to:
     *
     * <pre>
     * P(1,x) = 1 - e^{-x}
     * </pre>
     *
     * <p>This identity is particularly useful because it provides an exact
     * reference function that is easy to compute and does not depend on the
     * internal implementation of the gamma utilities.
     *
     * <p>Passing this test strongly suggests that:
     * <ul>
     *   <li>the normalization is correct,</li>
     *   <li>the exponential term is correct, and</li>
     *   <li>the numerical approximation behaves properly over a range of
     *       arguments.</li>
     * </ul>
     */
    @Test
    @DisplayName("For a = 1, P(1,x) = 1 - e^(-x)")
    void regularizedGammaPForAEqualsOneShouldMatchClosedForm() {
        double[] xs = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0};

        for (double x : xs) {
            double expected = 1.0 - Math.exp(-x);
            double actual = StatisticalUtilities.regularizedGammaP(1.0, x);
            assertEquals(expected, actual, LOOSE_TOLERANCE,
                    "Failed for x=" + x);
        }
    }

    /**
     * Verifies the matching closed-form special case for the upper regularized
     * incomplete gamma function.
     *
     * <p>For {@code a = 1}, the upper regularized incomplete gamma function
     * reduces to:
     *
     * <pre>
     * Q(1,x) = e^{-x}
     * </pre>
     *
     * <p>This is the complement of the previous identity and provides a very
     * strong independent validation of the upper-tail implementation.
     *
     * <p>In combination with the previous test, this also implicitly checks
     * that:
     *
     * <pre>
     * P(1,x) + Q(1,x) = 1
     * </pre>
     */
    @Test
    @DisplayName("For a = 1, Q(1,x) = e^(-x)")
    void regularizedGammaQForAEqualsOneShouldMatchClosedForm() {
        double[] xs = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0};

        for (double x : xs) {
            double expected = Math.exp(-x);
            double actual = StatisticalUtilities.regularizedGammaQ(1.0, x);
            assertEquals(expected, actual, LOOSE_TOLERANCE,
                    "Failed for x=" + x);
        }
    }

    /**
     * Verifies a closed-form special case of the chi-square cumulative
     * distribution function.
     *
     * <p>For a chi-square random variable with two degrees of freedom,
     * the CDF simplifies to:
     *
     * <pre>
     * F(x;2) = 1 - e^{-x/2}
     * </pre>
     *
     * <p>This follows from the fact that a chi-square distribution with
     * two degrees of freedom is equivalent to an exponential distribution
     * with rate {@code 1/2}.
     *
     * <p>This test ensures that the mapping from chi-square CDF to the
     * regularized gamma function has been implemented correctly.
     */
    @Test
    @DisplayName("Chi-square CDF with 2 degrees of freedom matches 1 - e^(-x/2)")
    void chiSquareCDFWithTwoDegreesOfFreedomShouldMatchClosedForm() {
        double[] xs = {0.0, 0.5, 1.0, 2.0, 4.0, 10.0};

        for (double x : xs) {
            double expected = 1.0 - Math.exp(-x / 2.0);
            double actual = StatisticalUtilities.chiSquareCDF(x, 2.0);
            assertEquals(expected, actual, LOOSE_TOLERANCE,
                    "Failed for x=" + x);
        }
    }

    /**
     * Verifies the corresponding closed-form expression for the right-tail
     * probability of a chi-square distribution with two degrees of freedom.
     *
     * <p>For {@code df = 2},
     *
     * <pre>
     * P(X ≥ x) = e^{-x/2}
     * </pre>
     *
     * <p>This test is especially important because the right-tail probability
     * is the quantity typically interpreted as a p-value in chi-square
     * hypothesis testing.
     *
     * <p>It helps detect semantic confusion between:
     * <ul>
     *   <li>the cumulative distribution function, and</li>
     *   <li>the upper-tail probability used in significance tests.</li>
     * </ul>
     */
    @Test
    @DisplayName("Chi-square p-value with 2 degrees of freedom matches e^(-x/2)")
    void chiSquarePValueWithTwoDegreesOfFreedomShouldMatchClosedForm() {
        double[] xs = {0.0, 0.5, 1.0, 2.0, 4.0, 10.0};

        for (double x : xs) {
            double expected = Math.exp(-x / 2.0);
            double actual = StatisticalUtilities.chiSquarePValue(x, 2.0);
            assertEquals(expected, actual, LOOSE_TOLERANCE,
                    "Failed for x=" + x);
        }
    }


    /**
     * Verifies the complementarity relation between the chi-square cumulative
     * distribution function and the chi-square right-tail probability:
     *
     * <pre>
     * CDF(x,df) + pValue(x,df) = 1
     * </pre>
     *
     * <p>This is the chi-square analogue of the identity
     * {@code P(a,x) + Q(a,x) = 1} for regularized gamma functions.
     *
     * <p>This test is important because the chi-square methods are higher-level
     * wrappers around the gamma methods, and therefore it validates not only
     * the numerical routines but also the correctness of the parameter
     * transformation:
     *
     * <pre>
     * a = df / 2
     * x = statistic / 2
     * </pre>
     */
    @Test
    @DisplayName("Chi-square CDF and p-value add up to 1")
    void chiSquareCDFAndPValueShouldAddUpToOne() {
        double[][] cases = {
                {0.0, 1.0},
                {0.1, 1.0},
                {1.0, 1.0},
                {2.0, 1.0},
                {0.0, 2.0},
                {1.0, 2.0},
                {5.0, 2.0},
                {3.0, 4.0},
                {10.0, 6.0},
                {20.0, 10.0}
        };

        for (double[] testCase : cases) {
            double statistic = testCase[0];
            double df = testCase[1];

            double cdf = StatisticalUtilities.chiSquareCDF(statistic, df);
            double pValue = StatisticalUtilities.chiSquarePValue(statistic, df);

            assertEquals(1.0, cdf + pValue, LOOSE_TOLERANCE,
                    "Failed for statistic=" + statistic + ", df=" + df);
        }
    }

    /**
     * Verifies the exact behaviour of the chi-square distribution at the
     * origin.
     *
     * <p>For any positive number of degrees of freedom:
     *
     * <pre>
     * CDF(0,df) = 0
     * pValue(0,df) = 1
     * </pre>
     *
     * <p>This test confirms that the implementation handles the boundary
     * statistic correctly and does not produce small spurious nonzero values
     * due to numerical approximation or floating-point issues.
     */
    @Test
    @DisplayName("Chi-square at zero: CDF = 0 and p-value = 1")
    void chiSquareAtZeroShouldReturnExpectedValues() {
        assertEquals(0.0, StatisticalUtilities.chiSquareCDF(0.0, 1.0), TOLERANCE);
        assertEquals(1.0, StatisticalUtilities.chiSquarePValue(0.0, 1.0), TOLERANCE);

        assertEquals(0.0, StatisticalUtilities.chiSquareCDF(0.0, 5.0), TOLERANCE);
        assertEquals(1.0, StatisticalUtilities.chiSquarePValue(0.0, 5.0), TOLERANCE);
    }

    /**
     * Regression test designed specifically to detect an accidental semantic
     * inversion between chi-square p-values and chi-square CDF values.
     *
     * <p>If a right-tail p-value is mistakenly implemented as a cumulative
     * probability, then this test will fail in an obvious way:
     *
     * <ul>
     *   <li>at statistic {@code 0}, the method would incorrectly return
     *       {@code 0} instead of {@code 1};</li>
     *   <li>at a very large statistic, the method would incorrectly return a
     *       value close to {@code 1} instead of a value close to {@code 0}.</li>
     * </ul>
     *
     * <p>This test is intentionally simple and highly diagnostic, making it
     * useful as a regression guard during refactoring.
     */
    @Test
    @DisplayName("Chi-square p-value regression: zero statistic gives 1, large statistic gives near 0")
    void chiSquarePValueRegressionShouldDetectCDFConfusion() {
        double pAtZero = StatisticalUtilities.chiSquarePValue(0.0, 4.0);
        double pAtLargeStatistic = StatisticalUtilities.chiSquarePValue(100.0, 4.0);

        assertEquals(1.0, pAtZero, TOLERANCE,
                "For chi-square statistic 0, the right-tail p-value must be 1.");

        assertTrue(pAtLargeStatistic < 1e-10,
                "For a very large chi-square statistic, the right-tail p-value must be near 0.");
    }

    /**
     * Regression test complementary to
     * {@link #chiSquarePValueRegressionShouldDetectCDFConfusion()}.
     *
     * <p>This test ensures that the chi-square cumulative distribution function
     * remains correctly interpreted as a left-tail probability:
     *
     * <ul>
     *   <li>at statistic {@code 0}, the CDF must be {@code 0};</li>
     *   <li>at a very large statistic, the CDF must approach {@code 1}.</li>
     * </ul>
     *
     * <p>Together, both regression tests provide a simple but robust semantic
     * safety net against confusion between lower-tail and upper-tail APIs.
     */
    @Test
    @DisplayName("Chi-square CDF regression: zero statistic gives 0, large statistic gives near 1")
    void chiSquareCDFRegressionShouldDetectPValueConfusion() {
        double cdfAtZero = StatisticalUtilities.chiSquareCDF(0.0, 4.0);
        double cdfAtLargeStatistic = StatisticalUtilities.chiSquareCDF(100.0, 4.0);

        assertEquals(0.0, cdfAtZero, TOLERANCE,
                "For chi-square statistic 0, the CDF must be 0.");

        assertTrue(cdfAtLargeStatistic > 1.0 - 1e-10,
                "For a very large chi-square statistic, the CDF must be near 1.");
    }

    /**
     * Verifies the exact value:
     *
     * <pre>
     * ln(Gamma(1)) = ln(1) = 0
     * </pre>
     *
     * <p>This is the simplest exact checkpoint for the logarithm of the gamma
     * function and serves as a baseline validation of the approximation.
     */
    @Test
    @DisplayName("gammaLn(1) = 0")
    void gammaLnOfOneShouldBeZero() {
        assertEquals(0.0, StatisticalUtilities.gammaLn(1.0), LOOSE_TOLERANCE);
    }

    /**
     * Verifies:
     *
     * <pre>
     * Gamma(2) = 1!
     * ln(Gamma(2)) = ln(1) = 0
     * </pre>
     *
     * <p>This test checks consistency of the gamma logarithm with the factorial
     * identity at a second integer point.
     */
    @Test
    @DisplayName("gammaLn(2) = 0 because Gamma(2) = 1")
    void gammaLnOfTwoShouldBeZero() {
        assertEquals(0.0, StatisticalUtilities.gammaLn(2.0), LOOSE_TOLERANCE);
    }

    /**
     * Verifies:
     *
     * <pre>
     * Gamma(3) = 2!
     * ln(Gamma(3)) = ln(2)
     * </pre>
     *
     * <p>Using small integer arguments provides exact reference values that
     * are particularly suitable for testing gamma-related numerical code.
     */
    @Test
    @DisplayName("gammaLn(3) = ln(2) because Gamma(3) = 2")
    void gammaLnOfThreeShouldMatchLogTwo() {
        assertEquals(Math.log(2.0), StatisticalUtilities.gammaLn(3.0), LOOSE_TOLERANCE);
    }

    /**
     * Verifies:
     *
     * <pre>
     * Gamma(5) = 4! = 24
     * ln(Gamma(5)) = ln(24)
     * </pre>
     *
     * <p>This extends the validation of {@code gammaLn} to a larger integer
     * value and helps confirm that the approximation remains accurate beyond
     * the smallest factorial cases.
     */
    @Test
    @DisplayName("gammaLn(5) = ln(24) because Gamma(5) = 24")
    void gammaLnOfFiveShouldMatchLog24() {
        assertEquals(Math.log(24.0), StatisticalUtilities.gammaLn(5.0), LOOSE_TOLERANCE);
    }

    /**
     * Verifies that {@link StatisticalUtilities#regularizedGammaP(double, double)}
     * rejects invalid values of the shape parameter {@code a}.
     *
     * <p>The regularized incomplete gamma function is defined only for
     * {@code a > 0}. This test ensures that the method rejects:
     *
     * <ul>
     *   <li>zero,</li>
     *   <li>negative values,</li>
     *   <li>{@code NaN},</li>
     *   <li>positive infinity.</li>
     * </ul>
     *
     * <p>Explicit domain validation is important because silent acceptance of
     * such values could produce meaningless numerical results and make debugging
     * much harder.
     */
    @Test
    @DisplayName("regularizedGammaP rejects invalid a")
    void regularizedGammaPShouldRejectInvalidA() {
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.regularizedGammaP(0.0, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.regularizedGammaP(-1.0, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.regularizedGammaP(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.regularizedGammaP(Double.POSITIVE_INFINITY, 1.0));
    }

    /**
     * Verifies that {@link StatisticalUtilities#regularizedGammaP(double, double)}
     * rejects invalid values of the argument {@code x}.
     *
     * <p>The lower regularized incomplete gamma function requires
     * {@code x >= 0}. This test checks that negative values and non-finite
     * values are rejected.
     */@Test
    @DisplayName("regularizedGammaP rejects invalid x")
    void regularizedGammaPShouldRejectInvalidX() {
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.regularizedGammaP(1.0, -1.0));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.regularizedGammaP(1.0, Double.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.regularizedGammaP(1.0, Double.POSITIVE_INFINITY));
    }

    /**
     * Verifies that {@link StatisticalUtilities#regularizedGammaQ(double, double)}
     * rejects invalid values of the shape parameter {@code a}.
     *
     * <p>This mirrors the domain checks for the lower regularized gamma function
     * and ensures consistent API behaviour between both complementary methods.
     */
    @Test
    @DisplayName("regularizedGammaQ rejects invalid a")
    void regularizedGammaQShouldRejectInvalidA() {
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.regularizedGammaQ(0.0, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.regularizedGammaQ(-1.0, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.regularizedGammaQ(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.regularizedGammaQ(Double.POSITIVE_INFINITY, 1.0));
    }

    /**
     * Verifies that {@link StatisticalUtilities#regularizedGammaQ(double, double)}
     * rejects invalid values of the argument {@code x}.
     *
     * <p>The upper regularized incomplete gamma function is defined for
     * {@code x >= 0}. This test confirms that invalid values do not pass
     * silently through the implementation.
     */
    @Test
    @DisplayName("regularizedGammaQ rejects invalid x")
    void regularizedGammaQShouldRejectInvalidX() {
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.regularizedGammaQ(1.0, -0.1));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.regularizedGammaQ(1.0, Double.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.regularizedGammaQ(1.0, Double.POSITIVE_INFINITY));
    }


    /**
     * Verifies that the chi-square cumulative distribution function rejects
     * invalid arguments.
     *
     * <p>The method requires:
     *
     * <ul>
     *   <li>{@code statistic >= 0}</li>
     *   <li>{@code degreesOfFreedom > 0}</li>
     * </ul>
     *
     * <p>This test also ensures rejection of non-finite values. These checks
     * are essential because chi-square probabilities are frequently used by
     * higher-level statistical procedures, and invalid inputs should fail fast.
     */
    @Test
    @DisplayName("chiSquareCDF rejects invalid arguments")
    void chiSquareCDFShouldRejectInvalidArguments() {
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.chiSquareCDF(-1.0, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.chiSquareCDF(1.0, 0.0));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.chiSquareCDF(1.0, -1.0));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.chiSquareCDF(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.chiSquareCDF(1.0, Double.NaN));
    }

    /**
     * Verifies that the chi-square right-tail probability rejects invalid
     * arguments under the same domain constraints as the chi-square CDF.
     *
     * <p>This test complements
     * {@link #chiSquareCDFShouldRejectInvalidArguments()} and ensures consistent
     * behaviour across both chi-square APIs.
     */
    @Test
    @DisplayName("chiSquarePValue rejects invalid arguments")
    void chiSquarePValueShouldRejectInvalidArguments() {
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.chiSquarePValue(-1.0, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.chiSquarePValue(1.0, 0.0));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.chiSquarePValue(1.0, -1.0));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.chiSquarePValue(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> StatisticalUtilities.chiSquarePValue(1.0, Double.NaN));
    }


    /**
     * Verifies backward compatibility of the deprecated
     * {@link StatisticalUtilities#chiSquare(double, double)} method.
     *
     * <p>In the current design, the deprecated method is expected to remain
     * equivalent to the chi-square cumulative distribution function, not to the
     * right-tail p-value.
     *
     * <p>This test exists to document and preserve that behaviour explicitly,
     * avoiding ambiguity during future maintenance.
     */
    @Test
    @DisplayName("Deprecated chiSquare method remains equivalent to chiSquareCDF")
    void deprecatedChiSquareShouldMatchChiSquareCDF() {
        double[][] cases = {
                {0.0, 1.0},
                {0.5, 1.0},
                {1.0, 2.0},
                {4.0, 2.0},
                {5.0, 4.0},
                {10.0, 8.0}
        };

        for (double[] testCase : cases) {
            double statistic = testCase[0];
            double df = testCase[1];

            double oldMethod = StatisticalUtilities.chiSquare(statistic, df);
            double cdf = StatisticalUtilities.chiSquareCDF(statistic, df);

            assertEquals(cdf, oldMethod, LOOSE_TOLERANCE,
                    "Failed for statistic=" + statistic + ", df=" + df);
        }
    }

    /**
     * Verifies that all probability-like outputs remain within the valid range
     * {@code [0,1]}.
     *
     * <p>This is a very important safety property for numerical code. Due to
     * rounding error, approximation error, or refactoring mistakes, numerical
     * routines can sometimes produce values slightly below {@code 0} or above
     * {@code 1}, even when the mathematical result is a probability.
     *
     * <p>This test checks that:
     *
     * <ul>
     *   <li>regularized gamma probabilities stay within bounds,</li>
     *   <li>chi-square cumulative probabilities stay within bounds,</li>
     *   <li>chi-square right-tail probabilities stay within bounds.</li>
     * </ul>
     */
    @Test
    @DisplayName("Probabilities stay within [0, 1]")
    void probabilitiesShouldStayWithinBounds() {
        double[][] gammaCases = {
                {0.5, 0.1},
                {0.5, 10.0},
                {1.0, 1.0},
                {2.0, 5.0},
                {10.0, 20.0},
                {50.0, 45.0}
        };

        for (double[] testCase : gammaCases) {
            double a = testCase[0];
            double x = testCase[1];

            double p = StatisticalUtilities.regularizedGammaP(a, x);
            double q = StatisticalUtilities.regularizedGammaQ(a, x);

            assertProbability(p, "P(" + a + "," + x + ")");
            assertProbability(q, "Q(" + a + "," + x + ")");
        }

        double[][] chiCases = {
                {0.0, 1.0},
                {0.5, 1.0},
                {1.0, 2.0},
                {10.0, 4.0},
                {30.0, 10.0}
        };

        for (double[] testCase : chiCases) {
            double statistic = testCase[0];
            double df = testCase[1];

            double cdf = StatisticalUtilities.chiSquareCDF(statistic, df);
            double pValue = StatisticalUtilities.chiSquarePValue(statistic, df);

            assertProbability(cdf, "CDF(" + statistic + "," + df + ")");
            assertProbability(pValue, "pValue(" + statistic + "," + df + ")");
        }
    }


    /**
     * Verifies that a numerical value intended to represent a probability is
     * contained in the closed interval {@code [0,1]}.
     *
     * <p>If the value lies outside that range, the method throws an
     * {@link AssertionError} with a descriptive label that identifies the
     * failing quantity.
     *
     * @param value numerical result expected to represent a probability
     * @param label human-readable identifier used in the assertion message
     * @throws AssertionError if {@code value < 0.0} or {@code value > 1.0}
     */
    private static void assertProbability(double value, String label) {
        if (value < 0.0 || value > 1.0) {
            throw new AssertionError(label + " is outside [0,1]: " + value);
        }
    }
}