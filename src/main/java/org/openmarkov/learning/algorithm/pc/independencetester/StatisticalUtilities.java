/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.pc.independencetester;

/**
 * The {@code StatisticalUtilities} class provides methods for computing the incomplete gamma function,
 * the log-gamma function, and the probability associated with the chi-square distribution.
 */
public class StatisticalUtilities {
    
    private static final int MAX_ITERATIONS = 100;
    private static final double EPSILON = 3.0e-7;
    private static final double MIN_FLOAT = 1.0e-30;
    
    /**
     * Computes the cumulative probability of the chi-square distribution for a given test statistic.
     * This function returns the cumulative probability up to `statistic`, equivalent to the
     * cumulative distribution function (CDF) of the chi-square distribution.
     * <p>
     * Mathematically, it is expressed as:
     *
     * <pre>
     * P(X ≤ statistic) = Fχ²(statistic, degreesOfFreedom)
     * </pre>
     * <p>
     * where Fχ²(statistic, degreesOfFreedom) is the cumulative distribution function of the chi-square distribution.
     * It is computed using the regularized incomplete gamma function P(a, x), where:
     *
     * <pre>
     * a = degreesOfFreedom / 2
     * x = statistic / 2
     * </pre>
     * <p>
     * This function is useful in hypothesis testing, such as goodness-of-fit or independence tests.
     *
     * @param statistic        The chi-square test statistic obtained from a hypothesis test.
     * @param degreesOfFreedom The degrees of freedom of the chi-square distribution (must be greater than 0).
     * @return The cumulative probability P(X ≤ statistic)
     * @throws IllegalArgumentException If `degreesOfFreedom` is less than or equal to 0
     * @see #gammp(double, double) For the computation of the regularized incomplete gamma function.
     */
    static public double chiSquare(double statistic, double degreesOfFreedom) {
        if (degreesOfFreedom <= 0) {
            throw new IllegalArgumentException("Degrees of freedom of chiSquare must be greater than 0.");
        }
        return (gammp(degreesOfFreedom / 2.0, statistic / 2.0));
    }
    
    /**
     * Computes the incomplete gamma function P(a, x).
     * Depending on the relationship between 'x' and 'a', this method uses either
     * the series approximation (gser) or continued fraction approximation (gcf)
     * to compute P(a, x).
     *
     * @param a the shape parameter of the incomplete gamma function (must be > 0)
     * @param x the value at which the incomplete gamma function is evaluated (must be >= 0)
     * @return the computed value of the incomplete gamma function P(a, x)
     */
    static public double gammp(double a, double x) {
        if (x < 0.0) {
            throw new IllegalArgumentException("x (second argument of gammp) must be greater or equal to 0");
        }
        if (a <= 0.0) {
            throw new IllegalArgumentException("a (first argument of gammp) must be greater than 0");
        }
        if (x < (a + 1.0)) {
            return gser(a, x); // Use the series approximation
        }
        return gammaCI(a, x); // Use the continued fraction approximation
    }
    
    /**
     * Computes the series approximation of the incomplete gamma function P(a, x).
     * This method calculates the series approximation of P(a, x) for given values
     * of 'a' and 'x'. It iteratively computes the series until convergence criteria
     * are met or the maximum number of iterations (MAX_ITERATIONS) is reached.
     *
     * @param a the shape parameter of the incomplete gamma function (must be > 0)
     * @param x the value at which the incomplete gamma function is evaluated (must be >= 0)
     * @return the computed value of the series approximation of P(a, x)
     */
    static public double gser(double a, double x) {
        if (x < 0.0) {
            throw new IllegalArgumentException("x (second argument of gser) should be greater or equal to 0");
        }
        if (x == 0.0) {
            return 0.0;
        }
        double gammaLn = gammaLn(a);
        double gammser;
        
        double sum = 1.0 / a;
        double del = sum;
        double ap = a;
        for (int n = 1; n <= MAX_ITERATIONS; n++) {
            ap++;
            del *= x / ap;
            sum += del;
            if (Math.abs(del) < (Math.abs(sum) * EPSILON)) {
                gammser = sum * Math.exp(-x + a * Math.log(x) - gammaLn);
                return gammser;
            }
        }
        System.out.println("Convergence not reached in 'gser' after " + MAX_ITERATIONS + " iterations. Parameter 'a' too large, MAX_ITERATIONS too small in routine 'gser' in 'LogFactorial'.");
        gammser = sum * Math.exp(-x + a * Math.log(x) - gammaLn);
        return gammser;
    }
    
    /**
     * Computes the complementary incomplete gamma function Q(a, x).
     * This method approximates the value of Q(a, x), which represents the probability
     * that a gamma-distributed random variable with shape parameter 'a' exceeds 'x'.
     *
     * @param a the shape parameter of the incomplete gamma function
     * @param x the value at which the incomplete gamma function is evaluated
     * @return the computed value of the complementary incomplete gamma function Q(a, x)
     */
    static public double gammaCI(double a, double x) {
        double gammaLn = gammaLn(a);
        double b = x + 1.0 - a;
        double c = 1.0 / MIN_FLOAT;
        double d = 1.0 / b;
        double h = d;
        
        int i;
        for (i = 1; i <= MAX_ITERATIONS; i++) {
            double an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if (Math.abs(d) < MIN_FLOAT)
                d = MIN_FLOAT;
            c = b + an / c;
            if (Math.abs(c) < MIN_FLOAT)
                c = MIN_FLOAT;
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (Math.abs(del - 1.0) < EPSILON)
                break;
        }
        if (i > MAX_ITERATIONS)
            System.out.println("Convergence not reached in 'gcf' after " + MAX_ITERATIONS + " iterations. Parameter a is too large, MAX_ITERATIONS too small in routine gcf.");
        double gammcf = Math.exp(-x + a * Math.log(x) - gammaLn) * h;
        return gammcf;
    }
    
    /**
     * Computes the natural logarithm of the gamma function, ln(Gamma(xx)).
     * This method approximates the natural logarithm of the gamma function for a given
     * value of 'xx'. It uses a series approximation with coefficients stored in 'cof'
     * to compute the logarithm efficiently.
     *
     * @param xx the value for which the natural logarithm of the gamma function is computed (must be > 0)
     * @return the computed value of ln(Gamma(xx))
     * @throws IllegalArgumentException if 'xx' is non-positive
     */
    static public double gammaLn(double xx) {
        if (xx <= 0) {
            throw new IllegalArgumentException("xx must be greater than 0");
        }
        
        double[] cof = {
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
        
        for (int j = 0; j < cof.length; j++) {
            ser += cof[j] / ++y;
        }
        
        return -tmp + Math.log(2.5066282746310005 * ser / x);
    }
    
}
