/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.pc.independencetester;

import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.potential.PotentialRole;
import org.openmarkov.core.model.network.potential.TablePotential;

import java.util.ArrayList;
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
	 * @param nodeX           <code>Node</code> first variable.
	 * @param nodeY           <code>Node</code> second variable.
	 * @param adjacencySubset <code>List</code> of <code>Node</code>
	 *                        representing the separation set (i.e. the conditional set).
	 * @return the score obtained in the independence test.
	 */
	public double test(CaseDatabase caseDatabase, Node nodeX, Node nodeY, List<Node> adjacencySubset) {
		if (caseDatabase == null || nodeX == null || nodeY == null || adjacencySubset == null) {
			throw new IllegalArgumentException("Input arguments in CrossEntropyIndependenceTester cannot be null.");
		}

		long degreesOfFreedom, numStatesAdjacency = 1, potentialSize = 1;
		double crossEntropy, chiS;
		List<Node> nodesYZ = new ArrayList<Node>();
		List<Node> nodesZ = new ArrayList<Node>();
		nodesYZ.add(nodeY);
		for (Node adjacent : adjacencySubset) {
			nodesYZ.add(adjacent);
			nodesZ.add(adjacent);
			numStatesAdjacency *= adjacent.getVariable().getNumStates();
		}
		potentialSize = numStatesAdjacency * nodeX.getVariable().getNumStates() * nodeY.getVariable().getNumStates();
		crossEntropy = crossEntropy(caseDatabase, nodeX, nodeY, nodesYZ, nodesZ);
		chiS = 2.0 * caseDatabase.getNumCases() * crossEntropy;
		chiS = (Math.abs(chiS) < 1e-10) ? 0.0 : chiS;

		degreesOfFreedom = numStatesAdjacency * (nodeX.getVariable().getNumStates() - 1) * (
				nodeY.getVariable().getNumStates() - 1
		);

		if (potentialSize < degreesOfFreedom)
			degreesOfFreedom = potentialSize;
		if (degreesOfFreedom <= 0)
			degreesOfFreedom = 1;

		return StatisticalUtilities.chiSquare(chiS, degreesOfFreedom);
	}

	/**
	 * Method that calculates the cross entropy between two nodes given a
	 * conditional set. We use the formula: CE(X,Y|Z) = H(X|Z) - H(X|Y,Z) (where
	 * CE means 'cross entropy' and H means 'entropy'.
	 *
	 * @param caseDatabase The database of cases
	 * @param nodeX The first variable
	 * @param nodeY The second variable
	 * @param nodesYZ The set of nodes Y and Z
	 * @param nodesZ  The conditional set Z
	 * @return the cross entropy between the two nodes X and Y given the conditional set Z
	 */
	private double crossEntropy(CaseDatabase caseDatabase, Node nodeX, Node nodeY, List<Node> nodesYZ,
			List<Node> nodesZ) {
		return (
				conditionedEntropy(caseDatabase, nodeX, nodesZ) - conditionedEntropy(caseDatabase, nodeX, nodesYZ)
		);
	}

	/**
	 * Method that calculates the conditioned entropy of a node given a
	 * conditional set.
	 *
	 * @param caseDatabase The database of cases
	 * @param nodeX        The target variable
	 * @param adjacencySubset The conditional set
	 * @return The conditioned entropy
	 */
	private double conditionedEntropy(CaseDatabase caseDatabase, Node nodeX, List<Node> adjacencySubset) {
		int numCases = caseDatabase.getNumCases();
		int numStates = nodeX.getVariable().getNumStates();

		// Calculate frequencies in the database
		List<Node> nodeAndAdjacency = new ArrayList<Node>();
		nodeAndAdjacency.add(nodeX);
		nodeAndAdjacency.addAll(adjacencySubset);
		TablePotential absoluteFreqPotential = absoluteFrequencies(caseDatabase, nodeAndAdjacency);
		double[] freq = absoluteFreqPotential.values;

		// Normalize
		for (int i = 0; i < freq.length; i++) {
			freq[i] /= numCases;
		}

		// Calculate entropy
		double nodeEntropy = 0;
		for (int j = 0; j < freq.length; j += numStates) {
			double n_ij = 0;
			for (int k = 0; k < numStates; k++)
				n_ij += freq[j + k];
			for (int k = 0; k < numStates; k++) {
				double n_ijk = freq[j + k];
				if (n_ijk > 0) {
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
	 * @param nodeList     <code>List</code> formed by the node and its parents.
	 * @return <code>TablePotential</code> with the absolute frequencies in the
	 * database of each of the configurations of the given node and its
	 * parents.
	 */
	private TablePotential absoluteFrequencies(CaseDatabase caseDatabase, List<Node> nodeList) {
		int index = 0;
		List<Variable> variables = new ArrayList<>();
		int[] indexes = new int[nodeList.size()];
		for (Node node : nodeList) {
			variables.add(node.getVariable());
			indexes[index] = caseDatabase.getVariables().indexOf(node.getVariable());
			index++;
		}

		TablePotential absoluteFreqPotential = new TablePotential(new ArrayList<Variable>(variables),
				PotentialRole.CONDITIONAL_PROBABILITY);
		double[] absoluteFreqs = absoluteFreqPotential.getValues();
		int[] offsets = absoluteFreqPotential.getOffsets();
		// Initialise the table
		for (int i = 0; i < absoluteFreqs.length; i++) {
			absoluteFreqs[i] = 0;
		}
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
