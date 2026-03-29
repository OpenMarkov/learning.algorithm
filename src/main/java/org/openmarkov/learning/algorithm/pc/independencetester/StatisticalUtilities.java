/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.pc.independencetester;

/**
 * Utility methods for the gamma function and the chi-square distribution.
 *
 * <p>This class provides:
 * <ul>
 *   <li>Regularized lower incomplete gamma function: P(a, x)</li>
 *   <li>Regularized upper incomplete gamma function: Q(a, x) = 1 - P(a, x)</li>
 *   <li>Chi-square cumulative distribution function (CDF)</li>
 *   <li>Chi-square right-tail probability (p-value)</li>
 * </ul>
 *
 * <p>For a chi-square random variable X with k degrees of freedom:
 *
 * <pre>
 * CDF(x; k)     = P(X ≤ x) = P(k/2, x/2)
 * p-value(x; k) = P(X ≥ x) = Q(k/2, x/2)
 * </pre>
 */
public final class StatisticalUtilities {

    private static final int MAX_ITERATIONS = 100;
    private static final double EPSILON = 3.0e-7;
    private static final double MIN_FLOAT = 1.0e-30;

    private StatisticalUtilities() {
        // Utility class: do not instantiate.
    }

    /**
     * Computes the chi-square cumulative distribution function (CDF).
     *
     * <pre>
     * P(X ≤ statistic) where X ~ ChiSquare(degreesOfFreedom)
     * </pre>
     *
     * @param statistic chi-square statistic value (must be >= 0)
     * @param degreesOfFreedom degrees of freedom (must be > 0)
     * @return chi-square CDF at {@code statistic}
     * @throws IllegalArgumentException if arguments are outside their domain
     * @throws ArithmeticException if the internal numerical method does not converge
     */
    public static double chiSquareCDF(double statistic, double degreesOfFreedom) {
        validateChiSquareArguments(statistic, degreesOfFreedom);
        return regularizedGammaP(degreesOfFreedom / 2.0, statistic / 2.0);
    }

    /**
     * Computes the right-tail probability (p-value) for the chi-square distribution.
     *
     * <pre>
     * P(X ≥ statistic) where X ~ ChiSquare(degreesOfFreedom)
     * </pre>
     *
     * <p>This is the value usually used in chi-square hypothesis tests.
     *
     * @param statistic chi-square statistic value (must be >= 0)
     * @param degreesOfFreedom degrees of freedom (must be > 0)
     * @return chi-square right-tail probability (p-value)
     * @throws IllegalArgumentException if arguments are outside their domain
     * @throws ArithmeticException if the internal numerical method does not converge
     */
    public static double chiSquarePValue(double statistic, double degreesOfFreedom) {
        validateChiSquareArguments(statistic, degreesOfFreedom);
        return regularizedGammaQ(degreesOfFreedom / 2.0, statistic / 2.0);
    }

    /**
     * Backward-compatible method.
     *
     * <p>Historically this method returned the chi-square CDF, not the p-value.
     * Prefer using {@link #chiSquareCDF(double, double)} or
     * {@link #chiSquarePValue(double, double)} explicitly.
     *
     * @param statistic chi-square statistic value (must be >= 0)
     * @param degreesOfFreedom degrees of freedom (must be > 0)
     * @return chi-square CDF at {@code statistic}
     * @deprecated Use {@link #chiSquareCDF(double, double)} or
     *             {@link #chiSquarePValue(double, double)}.
     */
    @Deprecated
    public static double chiSquare(double statistic, double degreesOfFreedom) {
        return chiSquareCDF(statistic, degreesOfFreedom);
    }

    /**
     * Computes the regularized lower incomplete gamma function P(a, x).
     *
     * @param a shape parameter (must be > 0)
     * @param x evaluation point (must be >= 0)
     * @return P(a, x)
     * @throws IllegalArgumentException if arguments are outside their domain
     * @throws ArithmeticException if the internal numerical method does not converge
     */
    public static double regularizedGammaP(double a, double x) {
        validateGammaArguments(a, x);

        if (x == 0.0) {
            return 0.0;
        }

        if (x < a + 1.0) {
            return clampProbability(gser(a, x));
        }

        return clampProbability(1.0 - gammaCF(a, x));
    }

    /**
     * Computes the regularized upper incomplete gamma function Q(a, x).
     *
     * @param a shape parameter (must be > 0)
     * @param x evaluation point (must be >= 0)
     * @return Q(a, x) = 1 - P(a, x)
     * @throws IllegalArgumentException if arguments are outside their domain
     * @throws ArithmeticException if the internal numerical method does not converge
     */
    public static double regularizedGammaQ(double a, double x) {
        validateGammaArguments(a, x);

        if (x == 0.0) {
            return 1.0;
        }

        if (x < a + 1.0) {
            return clampProbability(1.0 - gser(a, x));
        }

        return clampProbability(gammaCF(a, x));
    }

    /**
     * Series representation for the regularized lower incomplete gamma function P(a, x).
     * Best suited for x < a + 1.
     */
    private static double gser(double a, double x) {
        validateGammaArguments(a, x);

        if (x == 0.0) {
            return 0.0;
        }

        final double gammaLn = gammaLn(a);

        double sum = 1.0 / a;
        double del = sum;
        double ap = a;

        for (int n = 1; n <= MAX_ITERATIONS; n++) {
            ap += 1.0;
            del *= x / ap;
            sum += del;

            if (Math.abs(del) < Math.abs(sum) * EPSILON) {
                double value = sum * Math.exp(-x + a * Math.log(x) - gammaLn);
                return clampProbability(value);
            }
        }

        throw new ArithmeticException(
                "Convergence not reached in gser after " + MAX_ITERATIONS + " iterations.");
    }

    /**
     * Continued-fraction representation for the regularized upper incomplete gamma
     * function Q(a, x). Best suited for x >= a + 1.
     */
    private static double gammaCF(double a, double x) {
        validateGammaArguments(a, x);

        if (x == 0.0) {
            return 1.0;
        }

        final double gammaLn = gammaLn(a);

        double b = x + 1.0 - a;
        if (Math.abs(b) < MIN_FLOAT) {
            b = MIN_FLOAT;
        }

        double c = 1.0 / MIN_FLOAT;
        double d = 1.0 / b;
        double h = d;

        for (int i = 1; i <= MAX_ITERATIONS; i++) {
            double an = -i * (i - a);
            b += 2.0;

            d = an * d + b;
            if (Math.abs(d) < MIN_FLOAT) {
                d = MIN_FLOAT;
            }

            c = b + an / c;
            if (Math.abs(c) < MIN_FLOAT) {
                c = MIN_FLOAT;
            }

            d = 1.0 / d;
            double del = d * c;
            h *= del;

            if (Math.abs(del - 1.0) < EPSILON) {
                double value = Math.exp(-x + a * Math.log(x) - gammaLn) * h;
                return clampProbability(value);
            }
        }

        throw new ArithmeticException(
                "Convergence not reached in gammaCF after " + MAX_ITERATIONS + " iterations.");
    }

    /**
     * Computes ln(Gamma(xx)) using a classical Lanczos-style approximation.
     *
     * @param xx argument (must be > 0)
     * @return ln(Gamma(xx))
     * @throws IllegalArgumentException if xx &lt;= 0
     */
    public static double gammaLn(double xx) {
        if (xx <= 0.0) {
            throw new IllegalArgumentException("xx must be greater than 0.");
        }

        final double[] cof = {
                76.18009172947146,
                -86.50532032941677,
                24.01409824083091,
                -1.231739572450155,
                0.1208650973866179e-2,
                -0.5395239384953e-5
        };

        double x = xx;
        double y = xx;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.log(tmp);
        double ser = 1.000000000190015;

        for (double coefficient : cof) {
            y += 1.0;
            ser += coefficient / y;
        }

        return -tmp + Math.log(2.5066282746310005 * ser / x);
    }

    private static void validateChiSquareArguments(double statistic, double degreesOfFreedom) {
        if (Double.isNaN(statistic) || Double.isInfinite(statistic) || statistic < 0.0) {
            throw new IllegalArgumentException("Statistic must be finite and greater than or equal to 0.");
        }
        if (Double.isNaN(degreesOfFreedom) || Double.isInfinite(degreesOfFreedom) || degreesOfFreedom <= 0.0) {
            throw new IllegalArgumentException("Degrees of freedom must be finite and greater than 0.");
        }
    }

    private static void validateGammaArguments(double a, double x) {
        if (Double.isNaN(a) || Double.isInfinite(a) || a <= 0.0) {
            throw new IllegalArgumentException("Parameter 'a' must be finite and greater than 0.");
        }
        if (Double.isNaN(x) || Double.isInfinite(x) || x < 0.0) {
            throw new IllegalArgumentException("Parameter 'x' must be finite and greater than or equal to 0.");
        }
    }

    private static double clampProbability(double value) {
        if (value < 0.0) {
            return 0.0;
        }
        if (value > 1.0) {
            return 1.0;
        }
        return value;
    }
}