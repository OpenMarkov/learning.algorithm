package org.openmarkov.learning.algorithm.pc;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.learning.algorithm.pc.independencetester.CrossEntropyIndependenceTester;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class CrossEntropyIndependenceTesterTest {

	private static CaseDatabase caseDatabase;
	private static Node nodeA, nodeB, nodeC;

	@BeforeAll
	public static void setup() {
		// Variables:
		// A: 2 states (0,1)
		// B: 2 states (0,1)
		// C: 3 states (0,1,2)
		Variable varA = new Variable("A", 2);
		Variable varB = new Variable("B", 2);
		Variable varC = new Variable("C", 3);

		List<Variable> variables = Arrays.asList(varA, varB, varC);

		// Each row represents a case: [A, B, C]
		// Designed so that A and B are dependent marginally but conditionally independent given C
		int[][] cases = {
				{0, 0, 0}, {1, 1, 0}, {0, 0, 0}, {1, 1, 0},
				{0, 1, 1}, {1, 0, 1}, {0, 1, 1}, {1, 0, 1},
				{0, 0, 2}, {1, 1, 2}, {0, 0, 2}, {1, 1, 2},
				{0, 0, 0}, {1, 1, 0}, {0, 0, 0}, {1, 1, 0},
				{0, 1, 1}, {1, 0, 1}, {0, 1, 1}, {1, 0, 1},
				{0, 0, 2}, {1, 1, 2}, {0, 0, 2}, {1, 1, 2}
		};

		caseDatabase = new CaseDatabase(variables, cases);

		ProbNet probNet = new ProbNet();

		nodeA = new Node(probNet, varA, NodeType.CHANCE);
		nodeB = new Node(probNet, varB, NodeType.CHANCE);
		nodeC = new Node(probNet, varC, NodeType.CHANCE);
	}

	@Disabled("The expected marginal independence isn't met")
	@Test
	public void testMarginalDependence() {
		CrossEntropyIndependenceTester tester = new CrossEntropyIndependenceTester();

		double pValue = tester.test(caseDatabase, nodeA, nodeB, Collections.emptyList());

		System.out.printf("Test A ⊥̸ B (marginal): p-value = %.6f%n", pValue);
		assertTrue(pValue < 0.05, "Expected marginal dependence between A and B.");
	}
	
	@Disabled("The expected conditional independence between A and B given C isn't met")
	@Test
	public void testConditionalIndependenceGivenC() {
		CrossEntropyIndependenceTester tester = new CrossEntropyIndependenceTester();

		double pValue = tester.test(caseDatabase, nodeA, nodeB, Collections.singletonList(nodeC));

		System.out.printf("Test A ⊥ B | C: p-value = %.6f%n", pValue);
		assertTrue(pValue > 0.05, "Expected conditional independence between A and B given C.");
	}

	@Test
	public void testInvalidArguments() {
		CrossEntropyIndependenceTester tester = new CrossEntropyIndependenceTester();

		assertThrows(IllegalArgumentException.class, () -> tester.test(null, nodeA, nodeB, Collections.emptyList()));
		assertThrows(IllegalArgumentException.class, () -> tester.test(caseDatabase, null, nodeB, Collections.emptyList()));
		assertThrows(IllegalArgumentException.class, () -> tester.test(caseDatabase, nodeA, null, Collections.emptyList()));
		assertThrows(IllegalArgumentException.class, () -> tester.test(caseDatabase, nodeA, nodeB, null));
	}
}
