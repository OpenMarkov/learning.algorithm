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
import org.openmarkov.core.model.network.potential.UniformPotential;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Iñigo
 * @author Manuel Arias
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

	/**
	 * B-EM-Uniform: a conditional-probability node whose potential is a
	 * {@link UniformPotential} (the default when a node has no explicit CPT) must
	 * be learnable. {@code adaptNetwork} used to cast the UniformPotential to
	 * TablePotential — sibling types with no subtype relation — throwing
	 * ClassCastException before any inference ran.
	 * <p>
	 * Scope: this test isolates B-EM-Uniform only. It asserts that the cast no
	 * longer happens; it does NOT assert on the learned parameters, which remain
	 * affected by B-EM-Reuse / B-EM-Order until those are fixed.
	 */
	@Test
	public void parametricLearningWithUniformPotentialCptDoesNotThrow() {
		ProbNet net = new ProbNet();
		Variable A = new Variable("A", "a0", "a1");
		Variable B = new Variable("B", "b0", "b1");
		net.addNode(A, NodeType.CHANCE);
		net.addNode(B, NodeType.CHANCE);
		net.addLink(A, B, true);

		// A: explicit table potential (root); B: UNIFORM potential (default CPT)
		net.addPotential(new TablePotential(Arrays.asList(A), PotentialRole.CONDITIONAL_PROBABILITY));
		net.addPotential(new UniformPotential(Arrays.asList(B, A), PotentialRole.CONDITIONAL_PROBABILITY));

		List<Variable> variables = new ArrayList<>();
		variables.add(A);
		variables.add(B);
		int[][] cases = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
		CaseDatabase caseDatabase = new CaseDatabase(variables, cases);

		EMAlgorithm em = new EMAlgorithm(net, caseDatabase, 0.0);

		Assertions.assertDoesNotThrow(em::parametricLearning,
				"parametricLearning must accept a UniformPotential CPT without ClassCastException");
	}

	/**
	 * B-EM-Reuse: the E-step must use each case's own evidence. A single OBSERVED
	 * root node has a one-variable CPT (no parent to transpose), so B-EM-Order is
	 * irrelevant and this isolates the inference-reuse bug: EM must learn the
	 * empirical distribution of the node.
	 * <p>
	 * With the bug, the reused {@code HuginPropagation} returns the first case's
	 * joint for every case (ClusterPropagation caches cluster messages/posteriors
	 * under StorageLevel.FULL and never invalidates them across evidence changes),
	 * so the learned CPT collapses to a spike at the first case's value.
	 */
	@Test
	public void parametricLearningOnObservedRootLearnsEmpiricalDistribution()
			throws IncompatibleEvidenceException.EvidenceIsIncompatibleWithOther, NonProjectablePotentialException,
			CannotNormalizePotentialException, NotEvaluableNetworkException.NotApplicableNetwork,
			ConstraintViolatedException {
		ProbNet net = new ProbNet();
		Variable C = new Variable("C", "c0", "c1", "c2");
		net.addNode(C, NodeType.CHANCE);
		net.addPotential(new TablePotential(Arrays.asList(C), PotentialRole.CONDITIONAL_PROBABILITY));

		List<Variable> variables = new ArrayList<>();
		variables.add(C);
		// empirical P(C) = [1/6, 2/6, 3/6]
		int[][] cases = { { 0 }, { 1 }, { 1 }, { 2 }, { 2 }, { 2 } };
		CaseDatabase caseDatabase = new CaseDatabase(variables, cases);

		EMAlgorithm em = new EMAlgorithm(net, caseDatabase, 0.0);
		em.parametricLearning();

		TablePotential learnedC = (TablePotential) net.getNode(C).getPotentials().get(0);
		Assertions.assertArrayEquals(new double[] { 1.0 / 6, 2.0 / 6, 3.0 / 6 }, learnedC.getValues(), 1e-9,
				"EM must learn the empirical distribution of an observed root node (each case's evidence, "
						+ "not the first case's reused inference result)");
	}

	/**
	 * B-EM-Order: the joint probability returned by {@code getJointProbability} is
	 * laid out in the junction-tree cluster's variable order, which need not match
	 * the CPT's [child, parents] order. The M-step indexes it as if the child were
	 * the fastest-varying variable, so counts land in transposed cells.
	 * <p>
	 * Isolation: fully observed A(2)->B(3), asymmetric P(B|A) so a transposition is
	 * detectable, and every (A,B) combination present so alpha=0 stays MLE-safe
	 * (no structural zeros). A is a root (one-variable CPT) so it is order-immune
	 * and stays correct; the discriminating assertion is on B's CPT.
	 */
	@Test
	public void parametricLearningLearnsAsymmetricConditionalCpt()
			throws IncompatibleEvidenceException.EvidenceIsIncompatibleWithOther, NonProjectablePotentialException,
			CannotNormalizePotentialException, NotEvaluableNetworkException.NotApplicableNetwork,
			ConstraintViolatedException {
		ProbNet net = new ProbNet();
		Variable A = new Variable("A", "a0", "a1");
		Variable B = new Variable("B", "b0", "b1", "b2");
		net.addNode(A, NodeType.CHANCE);
		net.addNode(B, NodeType.CHANCE);
		net.addLink(A, B, true);
		net.addPotential(new TablePotential(Arrays.asList(A), PotentialRole.CONDITIONAL_PROBABILITY));
		net.addPotential(new TablePotential(Arrays.asList(B, A), PotentialRole.CONDITIONAL_PROBABILITY));

		List<Variable> variables = new ArrayList<>();
		variables.add(A);
		variables.add(B);
		// a0: b0x3,b1x1,b2x1 -> P(B|a0)=[3/5,1/5,1/5]; a1: b0x2,b1x1,b2x1 -> P(B|a1)=[2/4,1/4,1/4]
		int[][] cases = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 1 }, { 0, 2 }, { 1, 0 }, { 1, 0 }, { 1, 1 }, { 1, 2 } };
		CaseDatabase caseDatabase = new CaseDatabase(variables, cases);

		EMAlgorithm em = new EMAlgorithm(net, caseDatabase, 0.0);
		em.parametricLearning();

		TablePotential learnedB = (TablePotential) net.getNode(B).getPotentials().get(0);
		// CPT variables are [B, A] (B fastest): index = b + 3*a
		double[] expected = { 3.0 / 5, 1.0 / 5, 1.0 / 5, 2.0 / 4, 1.0 / 4, 1.0 / 4 };
		Assertions.assertArrayEquals(expected, learnedB.getValues(), 1e-9,
				"EM must fill the CPT in its own [child, parents] variable order; the joint returned by "
						+ "getJointProbability must be realigned to the CPT order, not indexed as-is");
	}

	/**
	 * B-EM-Empty: with an empty case database the E-step loop never runs, so the
	 * expected-counts map stays empty and the M-step dereferenced a null from it.
	 * There is nothing to learn from no data, so EM must return the network
	 * unchanged instead of throwing a NullPointerException.
	 */
	@Test
	public void parametricLearningWithEmptyDatabaseReturnsNetworkUnchanged() {
		ProbNet net = new ProbNet();
		Variable A = new Variable("A", "a0", "a1");
		net.addNode(A, NodeType.CHANCE);
		net.addPotential(new TablePotential(Arrays.asList(A), PotentialRole.CONDITIONAL_PROBABILITY));

		List<Variable> variables = new ArrayList<>();
		variables.add(A);
		int[][] cases = new int[0][];
		CaseDatabase caseDatabase = new CaseDatabase(variables, cases);

		EMAlgorithm em = new EMAlgorithm(net, caseDatabase, 0.0);

		ProbNet result = Assertions.assertDoesNotThrow(em::parametricLearning,
				"parametricLearning must not throw on an empty database");
		// Nothing learned: the CPT stays at its initial uniform value.
		TablePotential learnedA = (TablePotential) result.getNode(A).getPotentials().get(0);
		Assertions.assertArrayEquals(new double[] { 0.5, 0.5 }, learnedA.getValues(), 1e-9,
				"an empty database must leave the network's potentials unchanged");
	}

	/**
	 * MLE fragility (the original "B-EMnan"): with alpha=0 and a parent
	 * configuration that never occurs in the data, the M-step divides 0/0 and
	 * writes NaN, which the next iteration's inference propagates across the whole
	 * CPT. There is nothing to estimate for an unobserved parent configuration, so
	 * its column must be left at its current (uniform) value instead.
	 */
	@Test
	public void parametricLearningWithUnobservedParentConfigurationKeepsItUniform()
			throws IncompatibleEvidenceException.EvidenceIsIncompatibleWithOther, NonProjectablePotentialException,
			CannotNormalizePotentialException, NotEvaluableNetworkException.NotApplicableNetwork,
			ConstraintViolatedException {
		ProbNet net = new ProbNet();
		Variable A = new Variable("A", "a0", "a1");
		Variable B = new Variable("B", "b0", "b1", "b2");
		net.addNode(A, NodeType.CHANCE);
		net.addNode(B, NodeType.CHANCE);
		net.addLink(A, B, true);
		net.addPotential(new TablePotential(Arrays.asList(A), PotentialRole.CONDITIONAL_PROBABILITY));
		net.addPotential(new TablePotential(Arrays.asList(B, A), PotentialRole.CONDITIONAL_PROBABILITY));

		List<Variable> variables = new ArrayList<>();
		variables.add(A);
		variables.add(B);
		// A=a1 never observed; a0: b0x2, b1x1, b2x1 -> P(B|a0) = [1/2, 1/4, 1/4]
		int[][] cases = { { 0, 0 }, { 0, 0 }, { 0, 1 }, { 0, 2 } };
		CaseDatabase caseDatabase = new CaseDatabase(variables, cases);

		EMAlgorithm em = new EMAlgorithm(net, caseDatabase, 0.0);
		em.parametricLearning();

		TablePotential learnedB = (TablePotential) net.getNode(B).getPotentials().get(0);
		// order [B, A], index = b + 3*a: P(B|a0) empirical, P(B|a1) left uniform
		double[] expected = { 1.0 / 2, 1.0 / 4, 1.0 / 4, 1.0 / 3, 1.0 / 3, 1.0 / 3 };
		Assertions.assertArrayEquals(expected, learnedB.getValues(), 1e-9,
				"an unobserved parent configuration must keep its uniform column, not become NaN");
	}
}
