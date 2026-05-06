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
 * Independence tester based on the G² (log-likelihood ratio) statistic
 * with Laplace smoothing.
 *
 * <p>The G² statistic for testing {@code X ⊥ Y | Z} is:
 * <pre>
 *   G² = 2 · Σ_{x,y,z}  ñ(x,y,z) · ln[ ñ(x,y,z) · ñ(z) / (ñ(x,z) · ñ(y,z)) ]
 * </pre>
 *
 * <p>Laplace smoothing adds one pseudocount to every cell of the joint
 * contingency table before computing the statistic. The smoothed marginals
 * are derived consistently from the smoothed joint:
 * <ul>
 *   <li>ñ(x,y,z) = n(x,y,z) + 1</li>
 *   <li>ñ(x,z)   = n(x,z) + |Y|</li>
 *   <li>ñ(y,z)   = n(y,z) + |X|</li>
 *   <li>ñ(z)     = n(z)   + |X|·|Y|</li>
 * </ul>
 *
 * <p>Under the null hypothesis of conditional independence, G² is
 * asymptotically distributed as χ² with {@code df = |Z|·(|X|-1)·(|Y|-1)},
 * and this class returns the corresponding right-tail p-value.
 *
 * <p>Compared to {@link CrossEntropyIndependenceTester}, this tester is more
 * robust when some cells of the contingency table have zero or near-zero
 * counts, which commonly occurs with small datasets or variables that have
 * low-probability states. Laplace smoothing prevents the statistic from
 * being distorted by empty cells and stabilises the p-value in those cases.
 *
 * @see CrossEntropyIndependenceTester
 * @see StatisticalUtilities
 */
public class G2IndependenceTester implements IndependenceTester {

    /**
     * Tests whether {@code nodeX} and {@code nodeY} are conditionally
     * independent given {@code adjacencySubset}.
     *
     * @param caseDatabase    the database of observed cases
     * @param nodeX           first variable
     * @param nodeY           second variable
     * @param adjacencySubset conditioning set Z (may be empty)
     * @return right-tail p-value of the G² statistic under χ²; a value
     *         above the chosen significance level indicates that independence
     *         cannot be rejected
     */
    @Override
    public double test(@NotNull CaseDatabase caseDatabase, @NotNull Node nodeX,
                       @NotNull Node nodeY, @NotNull List<Node> adjacencySubset) {

        int numStatesX = nodeX.getVariable().getNumStates();
        int numStatesY = nodeY.getVariable().getNumStates();

        // Number of joint configurations of Z = Π_i |Z_i|
        long numStatesZ = 1;
        for (Node z : adjacencySubset) {
            numStatesZ *= z.getVariable().getNumStates();
        }

        // Joint counts n(x, y, z): X varies fastest, then Y, then the Z block
        List<Node> nodesXYZ = new ArrayList<>();
        nodesXYZ.add(nodeX);
        nodesXYZ.add(nodeY);
        nodesXYZ.addAll(adjacencySubset);
        double[] nxyz = absoluteFrequencies(caseDatabase, nodesXYZ).getValues();

        int cellsXY = numStatesX * numStatesY;
        double g2 = 0.0;

        for (int z = 0; z < numStatesZ; z++) {
            int zBase = z * cellsXY;

            // Smoothed n(z) = Σ_{x,y} n(x,y,z) + |X|·|Y|
            double nz_s = cellsXY; // |X|·|Y| pseudocounts
            for (int i = 0; i < cellsXY; i++) {
                nz_s += nxyz[zBase + i];
            }

            // Precompute smoothed n(x,z) for every x: Σ_y n(x,y,z) + |Y|
            double[] nxz_s = new double[numStatesX];
            for (int x = 0; x < numStatesX; x++) {
                nxz_s[x] = numStatesY; // |Y| pseudocounts
                for (int y = 0; y < numStatesY; y++) {
                    nxz_s[x] += nxyz[zBase + y * numStatesX + x];
                }
            }

            for (int y = 0; y < numStatesY; y++) {
                // Smoothed n(y,z) = Σ_x n(x,y,z) + |X|
                double nyz_s = numStatesX; // |X| pseudocounts
                for (int x = 0; x < numStatesX; x++) {
                    nyz_s += nxyz[zBase + y * numStatesX + x];
                }

                for (int x = 0; x < numStatesX; x++) {
                    double count = nxyz[zBase + y * numStatesX + x] + 1.0; // smoothed n(x,y,z)
                    g2 += count * Math.log(count * nz_s / (nxz_s[x] * nyz_s));
                }
            }
        }

        g2 = 2.0 * g2;
        if (Math.abs(g2) < 1.0e-10) g2 = 0.0;
        if (g2 < 0.0) g2 = 0.0; // guard against tiny negative values from floating-point noise

        // Degrees of freedom: |Z| · (|X|-1) · (|Y|-1)
        long degreesOfFreedom = numStatesZ * (numStatesX - 1) * (numStatesY - 1);
        long potentialSize = numStatesZ * (long) cellsXY;
        if (potentialSize < degreesOfFreedom) degreesOfFreedom = potentialSize;
        if (degreesOfFreedom <= 0) degreesOfFreedom = 1;

        return StatisticalUtilities.chiSquarePValue(g2, degreesOfFreedom);
    }

    /**
     * Computes the absolute frequency table for the given list of nodes over
     * the supplied case database.
     *
     * <p>The first variable in {@code nodeList} varies fastest in the returned
     * array, which matches the standard OpenMarkov {@link TablePotential} layout.
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
