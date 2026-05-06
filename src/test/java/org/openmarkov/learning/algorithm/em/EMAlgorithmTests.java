/*
 * Copyright (c) CISIAD, UNED, Spain,  2018. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.em;

import org.junit.jupiter.api.*;
import org.openmarkov.core.exception.*;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.potential.PotentialRole;
import org.openmarkov.core.model.network.potential.TablePotential;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Iñigo
 * <p>
 * Tests for the Expectation-Maximization algorithm
 */
@TestInstance(TestInstance.Lifecycle.PER_METHOD)
@SuppressWarnings("deprecation")
public class EMAlgorithmTests {

	private EMAlgorithm algorithm;

	/**
     */
	@BeforeEach public void setUp() {
		ProbNet probNet = new ProbNet();

		Variable C = new Variable("C", "C0", "C1", "C2");
		Variable X1 = new Variable("X1", "X10", "X11", "X12");
		Variable X2 = new Variable("X2", "X20", "X21", "X22");
		Variable X3 = new Variable("X3", "X30", "X31", "X32");

		probNet.addNode(C, NodeType.CHANCE);
		probNet.addNode(X1, NodeType.CHANCE);
		probNet.addNode(X2, NodeType.CHANCE);
		probNet.addNode(X3, NodeType.CHANCE);

		probNet.addLink(C, X1, true);
		probNet.addLink(C, X2, true);
		probNet.addLink(C, X3, true);

		// Build a dummy probNet to pass the variable list to the CaseDatabase constructor
		ProbNet probNetDB = new ProbNet();
		probNetDB.addNode(X1, NodeType.CHANCE);
		probNetDB.addNode(X2, NodeType.CHANCE);
		probNetDB.addNode(X3, NodeType.CHANCE);

		TablePotential x1Potential = new TablePotential(Arrays.asList(X1, C), PotentialRole.CONDITIONAL_PROBABILITY);
		TablePotential x2Potential = new TablePotential(Arrays.asList(X2, C), PotentialRole.CONDITIONAL_PROBABILITY);
		TablePotential x3Potential = new TablePotential(Arrays.asList(X3, C), PotentialRole.CONDITIONAL_PROBABILITY);
		TablePotential cPotential = new TablePotential(Arrays.asList(C), PotentialRole.CONDITIONAL_PROBABILITY);

		probNet.addPotential(x1Potential);
		probNet.addPotential(x2Potential);
		probNet.addPotential(x3Potential);
		probNet.addPotential(cPotential);

		int cases[][] = { { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 } };
		List<Variable> variables = new ArrayList<>();
		variables.add(X1);
		variables.add(X2);
		variables.add(X3);

		CaseDatabase caseDatabase = new CaseDatabase(variables, cases);

		algorithm = new EMAlgorithm(probNet, caseDatabase, 0.0);

	}
    
    @Test
    public void test() throws IncompatibleEvidenceException.EvidenceIsIncompatibleWithOther, NonProjectablePotentialException, CannotNormalizePotentialException, NotEvaluableNetworkException.NotApplicableNetwork, ConstraintViolatedException {
		ProbNet learnedNet = algorithm.parametricLearning();
		Assertions.assertNotNull(learnedNet);
		Assertions.assertEquals(4, learnedNet.getNumNodes());
	}
}
