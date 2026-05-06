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
 * Independence tester that selects the most appropriate test method
 * based on the observed data conditions (Cochran's rule).
 *
 * <p>The selection logic is as follows. For each non-empty stratum z of Z
 * the expected count of cell (x, y) is:
 * <pre>
 *   E(x, y, z) = n(x, z) · n(y, z) / n(z)
 * </pre>
 *
 * <ul>
 *   <li>If every expected cell count is ≥ {@value #COCHRAN_THRESHOLD}, the
 *       asymptotic {@link CrossEntropyIndependenceTester} (χ²) is used.</li>
 *   <li>If any expected count is below {@value #COCHRAN_THRESHOLD} and the
 *       table is 2×2 in every stratum, the {@link FisherTester} (exact test)
 *       is used.</li>
 *   <li>Otherwise (non-binary table with small expected counts), the
 *       {@link G2IndependenceTester} (G² with Laplace smoothing) is used;
 *       Laplace smoothing keeps the statistic well-defined even with empty
 *       cells.</li>
 * </ul>
 *
 * @see FisherTester
 * @see G2IndependenceTester
 * @see CrossEntropyIndependenceTester
 */
public class AdaptativeTester implements IndependenceTester {

    /**
     * Minimum expected cell count below which the asymptotic chi-square
     * approximation may be unreliable (Cochran's criterion).
     */
    private static final double COCHRAN_THRESHOLD = 5.0;

    private final FisherTester fisherTester = new FisherTester();
    private final G2IndependenceTester g2Tester = new G2IndependenceTester();
    private final CrossEntropyIndependenceTester chiSquareTester = new CrossEntropyIndependenceTester();

    /**
     * Tests whether {@code nodeX} and {@code nodeY} are conditionally
     * independent given {@code adjacencySubset}, automatically selecting the
     * most appropriate test.
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
        boolean is2x2 = (numStatesX == 2 && numStatesY == 2);
        boolean smallExpectedCountFound = false;

        outer:
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

            for (int x = 0; x < numStatesX; x++) {
                for (int y = 0; y < numStatesY; y++) {
                    double expected = nxz[x] * nyz[y] / nz;
                    if (expected < COCHRAN_THRESHOLD) {
                        smallExpectedCountFound = true;
                        break outer;
                    }
                }
            }
        }

        if (!smallExpectedCountFound) {
            return chiSquareTester.test(caseDatabase, nodeX, nodeY, adjacencySubset);
        } else if (is2x2) {
            return fisherTester.test(caseDatabase, nodeX, nodeY, adjacencySubset);
        } else {
            return g2Tester.test(caseDatabase, nodeX, nodeY, adjacencySubset);
        }
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
