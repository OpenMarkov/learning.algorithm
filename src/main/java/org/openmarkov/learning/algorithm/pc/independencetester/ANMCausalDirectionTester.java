/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.pc.independencetester;

import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;

/**
 * Causal direction tester based on the Additive Noise Model (ANM) for discrete variables.
 *
 * <p>For a candidate direction X → Y, this tester fits the regression function
 * {@code f(x) = argmax_y P̂(Y=y | X=x)} and computes the residual noise
 * {@code ε = (y − f(x) + |Y|) mod |Y|}. If X → Y is the true causal direction,
 * the noise ε should be independent of X. The independence is assessed with a
 * chi-square test on the (X, ε) contingency table.
 *
 * <p>A <em>higher</em> p-value indicates stronger evidence that ε ⊥ X, i.e.
 * that X → Y is the true causal direction.
 *
 * <p>Reference: Peters, J., Mooij, J., Janzing, D., &amp; Schölkopf, B. (2014).
 * Causal discovery with continuous additive noise models. JMLR, 15, 2009–2053.
 * (Discrete adaptation via RESIT.)
 */
public class ANMCausalDirectionTester implements CausalDirectionTester {

    /**
     * Returns a p-value for the hypothesis that {@code nodeX} causally precedes {@code nodeY}.
     *
     * <p>A high p-value means the noise residual ε is approximately independent of X,
     * which supports the X → Y direction.
     *
     * @param caseDatabase the case database
     * @param nodeX        the candidate cause node
     * @param nodeY        the candidate effect node
     * @return p-value in [0, 1]; higher values support X → Y
     */
    @Override
    public double testDirection(CaseDatabase caseDatabase, Node nodeX, Node nodeY) {
        int numX = nodeX.getVariable().getNumStates();
        int numY = nodeY.getVariable().getNumStates();

        int idxX = caseDatabase.getVariables().indexOf(nodeX.getVariable());
        int idxY = caseDatabase.getVariables().indexOf(nodeY.getVariable());

        if (idxX < 0 || idxY < 0) {
            return 0.0;
        }

        int[][] cases = caseDatabase.getCases();

        // Step 1: build joint count table countXY[x][y]
        int[][] countXY = new int[numX][numY];
        for (int[] row : cases) {
            int x = row[idxX];
            int y = row[idxY];
            if (x >= 0 && y >= 0) {
                countXY[x][y]++;
            }
        }

        // Step 2: compute modal regression f(x) = argmax_y count(x, y)
        int[] f = new int[numX];
        for (int x = 0; x < numX; x++) {
            int best = 0;
            int bestCount = -1;
            for (int y = 0; y < numY; y++) {
                if (countXY[x][y] > bestCount) {
                    bestCount = countXY[x][y];
                    best = y;
                }
            }
            f[x] = best;
        }

        // Step 3: build noise contingency table noiseXE[x][epsilon]
        // epsilon = (y - f(x) + numY) % numY
        int[][] noiseXE = new int[numX][numY];
        int totalCases = 0;
        for (int[] row : cases) {
            int x = row[idxX];
            int y = row[idxY];
            if (x >= 0 && y >= 0) {
                int epsilon = (y - f[x] + numY) % numY;
                noiseXE[x][epsilon]++;
                totalCases++;
            }
        }

        if (totalCases == 0) {
            return 0.0;
        }

        // Step 4: compute chi-square statistic on noiseXE
        int[] rowSum = new int[numX];
        int[] colSum = new int[numY];
        for (int x = 0; x < numX; x++) {
            for (int e = 0; e < numY; e++) {
                rowSum[x] += noiseXE[x][e];
                colSum[e] += noiseXE[x][e];
            }
        }

        double chiSquare = 0.0;
        for (int x = 0; x < numX; x++) {
            for (int e = 0; e < numY; e++) {
                double expected = (double) rowSum[x] * colSum[e] / totalCases;
                if (expected > 0.0) {
                    double diff = noiseXE[x][e] - expected;
                    chiSquare += diff * diff / expected;
                }
            }
        }

        // Step 5: compute p-value
        long df = (long) (numX - 1) * (numY - 1);
        if (df <= 0) {
            df = 1;
        }

        return StatisticalUtilities.chiSquarePValue(chiSquare, df);
    }
}
