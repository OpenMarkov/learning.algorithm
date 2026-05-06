/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.pc.independencetester;

import org.jetbrains.annotations.NotNull;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.potential.PotentialRole;
import org.openmarkov.core.model.network.potential.TablePotential;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Independence tester based on Fisher's exact test.
 *
 * <p>For binary X and Y, this tester computes an exact p-value for each
 * stratum z of the conditioning set Z using the hypergeometric distribution,
 * then combines the per-stratum p-values into a single global p-value via
 * Fisher's combination method:
 *
 * <pre>
 *   T = -2 · Σ_z ln(p_z)  ~  χ²(2·k)
 * </pre>
 *
 * where k is the number of non-empty Z strata.
 *
 * <p>For variables with more than two states this tester falls back to the
 * Pearson chi-square approximation, since exact enumeration for general r×c
 * tables (Freeman–Halton) is computationally prohibitive.
 *
 * <p>This tester is most appropriate when the dataset is small or when some
 * cells of the contingency table have very low counts, making asymptotic
 * chi-square approximations unreliable.
 *
 * @see AdaptativeTester
 * @see StatisticalUtilities
 */
public class FisherTester implements IndependenceTester {

    /**
     * Tests whether {@code nodeX} and {@code nodeY} are conditionally
     * independent given {@code adjacencySubset}.
     *
     * @param caseDatabase    the database of observed cases
     * @param nodeX           first variable
     * @param nodeY           second variable
     * @param adjacencySubset conditioning set Z (may be empty)
     * @return p-value; a value above the chosen significance level indicates
     *         that independence cannot be rejected
     */
    @Override
    public double test(@NotNull CaseDatabase caseDatabase, @NotNull Node nodeX,
                       @NotNull Node nodeY, @NotNull List<Node> adjacencySubset) {

        int numStatesX = nodeX.getVariable().getNumStates();
        int numStatesY = nodeY.getVariable().getNumStates();

        long numStatesZ = 1;
        for (Node z : adjacencySubset) {
            numStatesZ *= z.getVariable().getNumStates();
        }

        List<Node> nodesXYZ = new ArrayList<>();
        nodesXYZ.add(nodeX);
        nodesXYZ.add(nodeY);
        nodesXYZ.addAll(adjacencySubset);
        double[] nxyz = absoluteFrequencies(caseDatabase, nodesXYZ).getValues();

        int cellsXY = numStatesX * numStatesY;

        // For non-binary variables fall back to Pearson chi-square
        if (numStatesX > 2 || numStatesY > 2) {
            return pearsonChiSquarePValue(nxyz, numStatesX, numStatesY, numStatesZ);
        }

        // Combine exact per-stratum Fisher p-values using Fisher's method:
        // T = -2 * sum(ln(p_z)) ~ chi2(2 * nonEmptyStrata)
        double combinedStatistic = 0.0;
        int nonEmptyStrata = 0;

        for (int z = 0; z < numStatesZ; z++) {
            int base = z * cellsXY;
            // Layout: index(x, y, z) = x + y*numStatesX + z*cellsXY, X varies fastest
            int n00 = (int) nxyz[base];                    // x=0, y=0
            int n10 = (int) nxyz[base + 1];                // x=1, y=0
            int n01 = (int) nxyz[base + numStatesX];       // x=0, y=1
            int n11 = (int) nxyz[base + numStatesX + 1];   // x=1, y=1

            if (n00 + n10 + n01 + n11 == 0) continue;

            double pz = fisher2x2PValue(n00, n10, n01, n11);
            if (pz <= 0.0) pz = Double.MIN_VALUE; // guard against log(0)
            combinedStatistic += -2.0 * Math.log(pz);
            nonEmptyStrata++;
        }

        if (nonEmptyStrata == 0) return 1.0;
        if (combinedStatistic < 0.0) combinedStatistic = 0.0;

        // Fisher's combined statistic ~ chi2(2 * nonEmptyStrata)
        return StatisticalUtilities.chiSquarePValue(combinedStatistic, 2.0 * nonEmptyStrata);
    }

    /**
     * Two-tailed p-value for a 2×2 contingency table using Fisher's exact test.
     *
     * <p>Enumerates all tables with the same row and column margins as the
     * observed table and sums the hypergeometric probabilities of those that
     * are at most as likely as the observed one.
     *
     * @param a count n(X=0, Y=0)
     * @param b count n(X=1, Y=0)
     * @param c count n(X=0, Y=1)
     * @param d count n(X=1, Y=1)
     * @return two-tailed exact p-value
     */
    static double fisher2x2PValue(int a, int b, int c, int d) {
        int N  = a + b + c + d;
        int R0 = a + b; // first row margin  (Y=0)
        int C0 = a + c; // first column margin (X=0)

        double logPObs = logHypergeometric(a, R0, C0, N);

        double pValue = 0.0;
        int kMin = Math.max(0, R0 + C0 - N);
        int kMax = Math.min(R0, C0);

        for (int k = kMin; k <= kMax; k++) {
            double logPk = logHypergeometric(k, R0, C0, N);
            // Sum all tables at least as extreme (log-probability ≤ observed + tolerance)
            if (logPk <= logPObs + 1.0e-10) {
                pValue += Math.exp(logPk);
            }
        }
        return Math.min(pValue, 1.0);
    }

    /**
     * Log probability of a cell value {@code k} under the hypergeometric distribution:
     * <pre>
     *   log P(X = k) = log C(K, k) + log C(N−K, n−k) − log C(N, n)
     * </pre>
     *
     * @param k observed value
     * @param K number of success states in the population (row margin)
     * @param n number of draws (column margin)
     * @param N population size
     * @return log probability
     */
    private static double logHypergeometric(int k, int K, int n, int N) {
        return logBinomial(K, k) + logBinomial(N - K, n - k) - logBinomial(N, n);
    }

    /**
     * Natural logarithm of the binomial coefficient C(n, k),
     * computed via the log-gamma function for numerical stability.
     */
    private static double logBinomial(int n, int k) {
        if (k < 0 || k > n) return Double.NEGATIVE_INFINITY;
        if (k == 0 || k == n) return 0.0;
        return StatisticalUtilities.gammaLn(n + 1.0)
                - StatisticalUtilities.gammaLn(k + 1.0)
                - StatisticalUtilities.gammaLn(n - k + 1.0);
    }

    /**
     * Pearson chi-square p-value for r×c|Z tables.
     * Used as fallback when at least one of X or Y has more than two states.
     */
    private static double pearsonChiSquarePValue(double[] nxyz, int numStatesX,
                                                  int numStatesY, long numStatesZ) {
        int cellsXY = numStatesX * numStatesY;
        double chiSq = 0.0;

        for (int z = 0; z < numStatesZ; z++) {
            int base = z * cellsXY;
            double nz = 0.0;
            for (int i = 0; i < cellsXY; i++) nz += nxyz[base + i];
            if (nz == 0.0) continue;

            double[] nxz = new double[numStatesX];
            double[] nyz = new double[numStatesY];
            for (int y = 0; y < numStatesY; y++) {
                for (int x = 0; x < numStatesX; x++) {
                    double count = nxyz[base + y * numStatesX + x];
                    nxz[x] += count;
                    nyz[y] += count;
                }
            }

            for (int y = 0; y < numStatesY; y++) {
                for (int x = 0; x < numStatesX; x++) {
                    double expected = nxz[x] * nyz[y] / nz;
                    if (expected > 0.0) {
                        double obs = nxyz[base + y * numStatesX + x];
                        chiSq += (obs - expected) * (obs - expected) / expected;
                    }
                }
            }
        }

        long df = numStatesZ * (numStatesX - 1L) * (numStatesY - 1L);
        if (df <= 0) df = 1;
        return StatisticalUtilities.chiSquarePValue(chiSq, df);
    }

    /**
     * Computes the absolute frequency table for the given list of nodes over
     * the supplied case database.
     *
     * <p>The first variable in {@code nodeList} varies fastest in the returned
     * array, matching the standard OpenMarkov {@link TablePotential} layout.
     */
    private static TablePotential absoluteFrequencies(CaseDatabase caseDatabase, List<Node> nodeList) {
        int numNodes = nodeList.size();
        List<Variable> variables = new ArrayList<>();
        int[] indexes = new int[numNodes];
        for (int i = 0; i < numNodes; i++) {
            variables.add(nodeList.get(i).getVariable());
            indexes[i] = caseDatabase.getVariables().indexOf(nodeList.get(i).getVariable());
        }
        TablePotential pot = new TablePotential(new ArrayList<>(variables), PotentialRole.CONDITIONAL_PROBABILITY);
        double[] freqs = pot.getValues();
        int[] offsets = pot.getOffsets();
        Arrays.fill(freqs, 0.0);
        for (int[] row : caseDatabase.getCases()) {
            int idx = 0;
            for (int v = 0; v < numNodes; v++) {
                idx += offsets[v] * row[indexes[v]];
            }
            freqs[idx]++;
        }
        return pot;
    }
}
