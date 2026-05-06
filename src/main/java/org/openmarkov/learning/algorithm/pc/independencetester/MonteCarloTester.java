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

import java.util.List;
import java.util.Random;

/**
 * Independence tester based on a Monte Carlo permutation test.
 *
 * <p>The null distribution of the G² statistic is estimated empirically by
 * repeatedly permuting the observed data under the null hypothesis of
 * conditional independence:
 *
 * <ol>
 *   <li>Compute the observed G² statistic for X ⊥ Y | Z.</li>
 *   <li>For each of {@code numPermutations} replications:
 *     <ol type="a">
 *       <li>Permute the values of X <em>within each Z stratum</em>.
 *           This preserves the joint distributions of (X, Z) and (Y, Z)
 *           while breaking the conditional X–Y association given Z.</li>
 *       <li>Compute G² on the permuted data.</li>
 *     </ol>
 *   </li>
 *   <li>Return the Monte Carlo p-value using Phipson &amp; Smyth's correction:
 *       <pre>(B + 1) / (R + 1)</pre>
 *       where B is the number of permuted statistics ≥ the observed one and
 *       R is the total number of permutations. The +1 offset guarantees a
 *       valid (strictly positive) p-value.</li>
 * </ol>
 *
 * <p>This test makes no asymptotic assumptions and is therefore valid for any
 * sample size or variable cardinality, at the cost of additional computation.
 * The G² statistic used internally is the raw log-likelihood ratio
 * (without Laplace smoothing), since the permuted null distribution is
 * estimated empirically.
 *
 * @see G2IndependenceTester
 * @see StatisticalUtilities
 */
public class MonteCarloTester implements IndependenceTester {

    /** Default number of permutations. */
    public static final int DEFAULT_NUM_PERMUTATIONS = 999;

    /** Default random seed for reproducibility. */
    public static final long DEFAULT_SEED = 42L;

    private final int numPermutations;
    private final Random random;

    /**
     * Constructs a tester with {@value #DEFAULT_NUM_PERMUTATIONS} permutations
     * and a fixed seed of {@value #DEFAULT_SEED} for reproducibility.
     */
    public MonteCarloTester() {
        this(DEFAULT_NUM_PERMUTATIONS, DEFAULT_SEED);
    }

    /**
     * Constructs a tester with the given number of permutations and seed.
     *
     * @param numPermutations number of random permutations (must be ≥ 1)
     * @param seed            seed for the random number generator
     * @throws IllegalArgumentException if {@code numPermutations} &lt; 1
     */
    public MonteCarloTester(int numPermutations, long seed) {
        if (numPermutations < 1) {
            throw new IllegalArgumentException("numPermutations must be at least 1.");
        }
        this.numPermutations = numPermutations;
        this.random = new Random(seed);
    }

    /**
     * Tests whether {@code nodeX} and {@code nodeY} are conditionally
     * independent given {@code adjacencySubset} using a permutation test.
     *
     * @param caseDatabase    the database of observed cases
     * @param nodeX           first variable
     * @param nodeY           second variable
     * @param adjacencySubset conditioning set Z (may be empty)
     * @return Monte Carlo p-value; a value above the chosen significance level
     *         indicates that independence cannot be rejected
     */
    @Override
    public double test(@NotNull CaseDatabase caseDatabase, @NotNull Node nodeX,
                       @NotNull Node nodeY, @NotNull List<Node> adjacencySubset) {

        List<Variable> dbVars = caseDatabase.getVariables();
        int idxX = dbVars.indexOf(nodeX.getVariable());
        int idxY = dbVars.indexOf(nodeY.getVariable());

        int numZ = adjacencySubset.size();
        int[] idxZ      = new int[numZ];
        int[] numStatesZ = new int[numZ];
        for (int i = 0; i < numZ; i++) {
            idxZ[i]       = dbVars.indexOf(adjacencySubset.get(i).getVariable());
            numStatesZ[i] = adjacencySubset.get(i).getVariable().getNumStates();
        }

        int numStatesX = nodeX.getVariable().getNumStates();
        int numStatesY = nodeY.getVariable().getNumStates();

        // totalStatesZ = Π_i numStatesZ[i]  (1 when Z is empty)
        int totalStatesZ = 1;
        long[] zMultipliers = new long[numZ]; // offset per Z variable for stratum index
        for (int i = 0; i < numZ; i++) {
            zMultipliers[i] = totalStatesZ;
            totalStatesZ *= numStatesZ[i];
        }

        int[][] cases = caseDatabase.getCases();
        int n = cases.length;

        // Pre-extract X, Y values and stratum index per case
        int[] xVals    = new int[n];
        int[] yVals    = new int[n];
        int[] stratumOf = new int[n]; // Z stratum index for each case

        for (int i = 0; i < n; i++) {
            xVals[i] = cases[i][idxX];
            yVals[i] = cases[i][idxY];
            int zIdx = 0;
            for (int j = 0; j < numZ; j++) {
                zIdx += (int) (zMultipliers[j] * cases[i][idxZ[j]]);
            }
            stratumOf[i] = zIdx;
        }

        // Group case indices by Z stratum (for efficient permutation)
        int[] stratumSize = new int[totalStatesZ];
        for (int i = 0; i < n; i++) stratumSize[stratumOf[i]]++;

        int[][] stratumIndices = new int[totalStatesZ][];
        for (int z = 0; z < totalStatesZ; z++) stratumIndices[z] = new int[stratumSize[z]];
        int[] pos = new int[totalStatesZ];
        for (int i = 0; i < n; i++) {
            int z = stratumOf[i];
            stratumIndices[z][pos[z]++] = i;
        }

        // Compute observed G² statistic
        double observedG2 = computeG2(xVals, yVals, stratumOf,
                numStatesX, numStatesY, totalStatesZ);

        // Permutation loop: permute X within each Z stratum
        int[] permX = new int[n];
        int countExtreme = 0;

        for (int perm = 0; perm < numPermutations; perm++) {
            System.arraycopy(xVals, 0, permX, 0, n);
            for (int z = 0; z < totalStatesZ; z++) {
                shuffleWithin(permX, stratumIndices[z]);
            }
            double permG2 = computeG2(permX, yVals, stratumOf,
                    numStatesX, numStatesY, totalStatesZ);
            if (permG2 >= observedG2) countExtreme++;
        }

        // Phipson & Smyth correction: (B + 1) / (R + 1)
        return (countExtreme + 1.0) / (numPermutations + 1.0);
    }

    /**
     * Computes the raw G² statistic for X ⊥ Y | Z directly from pre-extracted
     * value arrays (no Laplace smoothing).
     *
     * <p>The statistic is:
     * <pre>
     *   G² = 2 · Σ_{x,y,z} n(x,y,z) · ln[ n(x,y,z) · n(z) / (n(x,z) · n(y,z)) ]
     * </pre>
     *
     * Zero-count cells are skipped (they contribute 0 to the sum).
     */
    private static double computeG2(int[] xVals, int[] yVals, int[] stratumOf,
                                     int numStatesX, int numStatesY, int totalStatesZ) {
        int cellsXY = numStatesX * numStatesY;
        double[] nxyz = new double[totalStatesZ * cellsXY];

        for (int i = 0; i < xVals.length; i++) {
            // index = x + y * numStatesX + z * cellsXY  (X varies fastest)
            nxyz[xVals[i] + yVals[i] * numStatesX + stratumOf[i] * cellsXY]++;
        }

        double g2 = 0.0;

        for (int z = 0; z < totalStatesZ; z++) {
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
                    double count = nxyz[base + y * numStatesX + x];
                    if (count > 0.0 && nxz[x] > 0.0 && nyz[y] > 0.0) {
                        g2 += count * Math.log(count * nz / (nxz[x] * nyz[y]));
                    }
                }
            }
        }

        return Math.max(0.0, 2.0 * g2);
    }

    /**
     * Performs a Fisher–Yates shuffle of the elements at the given indices
     * within the array {@code arr}, leaving all other positions unchanged.
     *
     * @param arr     the array to permute in place
     * @param indices the positions within {@code arr} to shuffle
     */
    private void shuffleWithin(int[] arr, int[] indices) {
        int k = indices.length;
        for (int i = k - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = arr[indices[i]];
            arr[indices[i]] = arr[indices[j]];
            arr[indices[j]] = tmp;
        }
    }
}
