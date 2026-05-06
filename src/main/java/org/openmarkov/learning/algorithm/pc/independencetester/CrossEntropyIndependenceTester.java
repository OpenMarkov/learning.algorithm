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
 * This class implements an independence tester based on the conditional entropy
 * criterion.
 *
 * @author joliva
 * @author ibermejo
 */
public class CrossEntropyIndependenceTester implements IndependenceTester {
    
    /**
     * This method computes the value of the independence test for two nodes
     *
     * @param nodeX           {@code Node} first variable.
     * @param nodeY           {@code Node} second variable.
     * @param adjacencySubset {@code List} of {@code Node}
     *                        representing the separation set (i.e. the conditional set).
     * @return the score obtained in the independence test.
     */
    @Override
    public double test(@NotNull CaseDatabase caseDatabase, @NotNull Node nodeX, @NotNull Node nodeY,
                       @NotNull List<Node> adjacencySubset) {
        long numStatesAdjacency = 1;
        double chiS;
        
        // nodesYZ = {Y, Z1, Z2, ..., Zn}
        // nodesZ  = {Z1, Z2, ..., Zn}
        List<Node> nodesYZ = new ArrayList<>();
        List<Node> nodesZ = new ArrayList<>();
        nodesYZ.add(nodeY);
        for (Node adjacent : adjacencySubset) {
            nodesYZ.add(adjacent);
            nodesZ.add(adjacent);
            // Compute the product of the number of states of all nodes in Z
            numStatesAdjacency *= adjacent.getVariable().getNumStates();
        }
        
        // Compute the joint state space size
        long potentialSize = numStatesAdjacency * nodeX.getVariable().getNumStates() * nodeY.getVariable()
                                                                                            .getNumStates();
        // Compute the cross-entropy between X and Y given Z
        double crossEntropy = crossEntropy(caseDatabase, nodeX, nodeY, nodesYZ, nodesZ);
        // Compute the chi-square statistic
        chiS = 2.0 * caseDatabase.getNumCases() * crossEntropy;
        // Prevent numerical instabilities near zero
        chiS = (Math.abs(chiS) < 1.0e-10) ? 0.0 : chiS;
        
        // Compute the effective degrees of freedom by summing (|X_z|-1)*(|Y_z|-1) over strata z,
        // where |X_z| and |Y_z| are the number of states of X and Y with non-zero marginal count
        // in stratum z.  This correctly handles degenerate strata where one variable is constant
        // (e.g., a deterministic OR node): those strata contribute 0 to the df, and if ALL strata
        // are degenerate the effective df is 0 and the test is undefined — we conservatively
        // declare dependence (return 0.0) rather than a spurious perfect-independence score.
        long degreesOfFreedom = computeEffectiveDf(
                caseDatabase, nodeX, nodeY, adjacencySubset, numStatesAdjacency);

        if (degreesOfFreedom <= 0) {
            // All strata have X or Y constant; the chi-square test is undefined here.
            // Conservative choice: declare dependence.
            return 0.0;
        }

        return StatisticalUtilities.chiSquarePValue(chiS, degreesOfFreedom); // Return the p-value
    }
    
    /**
     * Computes the effective degrees of freedom for the conditional independence test X ⊥ Y | Z.
     * <p>
     * The theoretical df = |Z| * (|X|-1) * (|Y|-1) overcounts when some strata of Z have X or Y
     * with fewer than their full number of states observed (e.g. a deterministic node that is
     * determined by its parents in Z).  The effective df sums (|X_z|-1)*(|Y_z|-1) over each
     * stratum z, where |X_z| and |Y_z| are the number of distinct values of X (resp. Y) that
     * appear in stratum z.  Strata where X or Y is constant contribute 0, so if every stratum
     * is degenerate the returned value is 0 (test undefined → caller should declare dependence).
     *
     * @param nodeX           first variable in the test
     * @param nodeY           second variable in the test
     * @param conditioningSet the conditioning set Z
     * @param numStatesZ      the total number of Z-combinations (product of state counts)
     * @return the effective degrees of freedom (0 if all strata are degenerate)
     */
    private long computeEffectiveDf(CaseDatabase caseDatabase, Node nodeX, Node nodeY,
                                    List<Node> conditioningSet, long numStatesZ) {
        int numStatesX = nodeX.getVariable().getNumStates();
        int numStatesY = nodeY.getVariable().getNumStates();

        // Build joint frequency table for (X, Y, Z1, ..., Zn).
        // Variable order: X fastest-varying, then Y, then Z (OpenMarkov little-endian convention).
        List<Node> xyzNodes = new ArrayList<>();
        xyzNodes.add(nodeX);
        xyzNodes.add(nodeY);
        xyzNodes.addAll(conditioningSet);
        TablePotential xyzTable = absoluteFrequencies(caseDatabase, xyzNodes);
        double[] freq = xyzTable.getValues();
        // freq[x + numStatesX*(y + numStatesY*z)] = n(X=x, Y=y, Z=z)

        long effectiveDf = 0;
        int blockSize = numStatesX * numStatesY;

        for (int z = 0; z < numStatesZ; z++) {
            int base = z * blockSize;

            // Count distinct X values observed in this stratum
            int xObservedStates = 0;
            for (int x = 0; x < numStatesX; x++) {
                double margXZ = 0;
                for (int y = 0; y < numStatesY; y++) {
                    margXZ += freq[base + x + numStatesX * y];
                }
                if (margXZ > 0) xObservedStates++;
            }

            // Count distinct Y values observed in this stratum
            int yObservedStates = 0;
            for (int y = 0; y < numStatesY; y++) {
                double margYZ = 0;
                for (int x = 0; x < numStatesX; x++) {
                    margYZ += freq[base + x + numStatesX * y];
                }
                if (margYZ > 0) yObservedStates++;
            }

            effectiveDf += (long) Math.max(0, xObservedStates - 1)
                         * (long) Math.max(0, yObservedStates - 1);
        }
        return effectiveDf;
    }

    /**
     * Method that calculates the cross entropy between two nodes given a
     * conditional set. We use the formula: CE(X,Y|Z) = H(X|Z) - H(X|Y,Z) (where H means 'entropy').
     *
     * @param caseDatabase The database of cases
     * @param nodeX        The first variable
     * @param nodeY        The second variable
     * @param nodesYZ      The set of nodes Y and Z
     * @param nodesZ       The conditional set Z
     * @return the cross entropy between the two nodes X and Y given the conditional set Z
     */
    private static double crossEntropy(CaseDatabase caseDatabase, Node nodeX, Node nodeY,
                                       List<Node> nodesYZ, List<Node> nodesZ) {
        return (
                conditionedEntropy(caseDatabase, nodeX, nodesZ) - conditionedEntropy(caseDatabase, nodeX, nodesYZ)
        );
    }
    
    /**
     * Computes the conditional entropy of a variable X given a set of conditioning variables Z.
     * <p>
     * The conditional entropy H(X | Z) is defined as:
     * </p>
     * <pre>
     *     H(X | Z) = -∑<sub>z</sub> ∑<sub>x</sub> p(x, z) · log(p(x | z))
     *              = -∑<sub>z</sub> ∑<sub>x</sub> p(x, z) · log(p(x, z) / p(z))
     * </pre>
     * <p>
     * where:
     * <ul>
     *   <li>p(x, z) is the joint probability of X = x and Z = z</li>
     *   <li>p(x | z) is the conditional probability of X = x given Z = z</li>
     * </ul>
     * </p>
     *
     * @param caseDatabase    The database of cases
     * @param nodeX           The target variable
     * @param adjacencySubset The conditional set
     * @return The conditioned entropy
     */
    private static double conditionedEntropy(CaseDatabase caseDatabase, Node nodeX, List<Node> adjacencySubset) {
        int numCases = caseDatabase.getNumCases();
        int numStates = nodeX.getVariable().getNumStates();
        
        // Construct the list of variables: [X, Z1, Z2, ..., Zn]
        List<Node> nodeAndAdjacency = new ArrayList<>();
        nodeAndAdjacency.add(nodeX);
        nodeAndAdjacency.addAll(adjacencySubset);
        
        // Compute the absolute joint frequency table for [X, Z]
        TablePotential absoluteFreqPotential = absoluteFrequencies(caseDatabase, nodeAndAdjacency);
        double[] freq = absoluteFreqPotential.getValues();
        
        // Normalize frequencies: convert counts to probabilities
        for (int i = 0; i < freq.length; i++) {
            freq[i] /= numCases;
        }
        
        // Calculate entropy
        double nodeEntropy = 0;
        // Iterate through each configuration of the conditioning set Z
        for (int j = 0; j < freq.length; j += numStates) {
            double n_ij = 0;
            // Sum the probabilities for each state of X given fixed Z = z_j
            for (int k = 0; k < numStates; k++)
                n_ij += freq[j + k];
            // Compute the contribution of each state x_k to H(X | Z = z_j)
            for (int k = 0; k < numStates; k++) {
                double n_ijk = freq[j + k];
                if (n_ijk > 0) {
                    // Contribution: p(x_k, z_j) * log(p(x_k | z_j))
                    nodeEntropy += n_ijk * Math.log(n_ijk / n_ij);
                }
            }
        }
        return -nodeEntropy;
    }
    
    /**
     * Calculate the absolute frequencies in the database of each of the
     * configurations of the given nodes.
     *
     * @param caseDatabase case database
     * @param nodeList     {@code List} formed by the node and its parents.
     * @return {@code TablePotential} with the absolute frequencies in the
     * database of each of the configurations of the given node and its
     * parents.
     */
    private static TablePotential absoluteFrequencies(CaseDatabase caseDatabase, List<Node> nodeList) {
        
        int numNodes = nodeList.size();
        List<Variable> variables = new ArrayList<>();
        int[] indexes = new int[numNodes];
        int index = 0;
        for (Node node : nodeList) {
            variables.add(node.getVariable());
            indexes[index] = caseDatabase.getVariables().indexOf(node.getVariable());
            index++;
        }
        
        TablePotential absoluteFreqPotential = new TablePotential(new ArrayList<>(variables),
                                                                  PotentialRole.CONDITIONAL_PROBABILITY);
        double[] absoluteFreqs = absoluteFreqPotential.getValues();
        int[] offsets = absoluteFreqPotential.getOffsets();
        Arrays.fill(absoluteFreqs, 0.0); // Initialice the table
        // Compute the absolute frequencies
        int numVariables = variables.size();
        int[][] cases = caseDatabase.getCases();
        for (int caseNumber = 0; caseNumber < cases.length; caseNumber++) {
            int indexCPT = 0;
            for (int variableNumber = 0; variableNumber < numVariables; ++variableNumber) {
                indexCPT += offsets[variableNumber] * cases[caseNumber][indexes[variableNumber]];
            }
            absoluteFreqs[indexCPT]++;
        }
        return absoluteFreqPotential;
    }
}
