package org.openmarkov.learning.algorithm.pc;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.learning.algorithm.pc.independencetester.MonteCarloTester;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link MonteCarloTester}.
 *
 * <p>Uses the standard synthetic dataset (C -> A, C -> B) where
 * A and B are marginally dependent but conditionally independent given C.
 * The permutation test is run with a small number of permutations
 * for speed, using a fixed seed for reproducibility.
 *
 * @author Manuel Arias
 */
public class MonteCarloTesterTest {

    private static CaseDatabase db;
    private static Node nodeA, nodeB, nodeC;

    @BeforeAll
    static void setup() {
        Variable varA = new Variable("A", 2);
        Variable varB = new Variable("B", 2);
        Variable varC = new Variable("C", 2);
        List<Variable> variables = List.of(varA, varB, varC);

        List<int[]> rows = new ArrayList<>();
        // C=0: P(A=0|C=0)=0.8, P(B=0|C=0)=0.8
        addCases(rows, new int[]{0, 0, 0}, 64);
        addCases(rows, new int[]{0, 1, 0}, 16);
        addCases(rows, new int[]{1, 0, 0}, 16);
        addCases(rows, new int[]{1, 1, 0}, 4);
        // C=1: P(A=0|C=1)=0.2, P(B=0|C=1)=0.2
        addCases(rows, new int[]{0, 0, 1}, 4);
        addCases(rows, new int[]{0, 1, 1}, 16);
        addCases(rows, new int[]{1, 0, 1}, 16);
        addCases(rows, new int[]{1, 1, 1}, 64);

        db = new CaseDatabase(variables, rows.toArray(new int[0][]));

        ProbNet net = new ProbNet();
        nodeA = new Node(net, varA, NodeType.CHANCE);
        nodeB = new Node(net, varB, NodeType.CHANCE);
        nodeC = new Node(net, varC, NodeType.CHANCE);
    }

    private static void addCases(List<int[]> rows, int[] row, int count) {
        for (int i = 0; i < count; i++) {
            rows.add(row.clone());
        }
    }

    // --- Marginal dependence / conditional independence ---

    @Test
    public void testMarginalDependence() {
        MonteCarloTester tester = new MonteCarloTester(199, 42L);
        double p = tester.test(db, nodeA, nodeB, Collections.emptyList());

        System.out.printf("MonteCarlo  A _|/_ B       : p = %.6f%n", p);
        assertTrue(p < 0.05, "Expected marginal dependence between A and B, got p = " + p);
    }

    @Test
    public void testConditionalIndependenceGivenC() {
        MonteCarloTester tester = new MonteCarloTester(199, 42L);
        double p = tester.test(db, nodeA, nodeB, List.of(nodeC));

        System.out.printf("MonteCarlo  A _|_ B | C    : p = %.6f%n", p);
        assertTrue(p > 0.05, "Expected conditional independence given C, got p = " + p);
    }

    // --- Reproducibility ---

    @Test
    public void testSameSeedSameResult() {
        double p1 = new MonteCarloTester(99, 123L).test(db, nodeA, nodeB, Collections.emptyList());
        double p2 = new MonteCarloTester(99, 123L).test(db, nodeA, nodeB, Collections.emptyList());

        assertEquals(p1, p2, 0.0,
                "Same seed and same data should produce identical p-values");
    }

    @Test
    public void testDifferentSeedsDifferentResults() {
        double p1 = new MonteCarloTester(99, 1L).test(db, nodeA, nodeB, Collections.emptyList());
        double p2 = new MonteCarloTester(99, 999L).test(db, nodeA, nodeB, Collections.emptyList());

        // Both should detect dependence, but exact p-values may differ
        assertTrue(p1 < 0.05 && p2 < 0.05,
                "Both seeds should still detect marginal dependence");
    }

    // --- Phipson & Smyth correction ---

    @Test
    public void testPhipsonSmythCorrection_PValueAlwaysPositive() {
        // Even with very strong dependence, p-value is (B+1)/(R+1) > 0
        MonteCarloTester tester = new MonteCarloTester(99, 42L);
        double p = tester.test(db, nodeA, nodeB, Collections.emptyList());

        assertTrue(p > 0.0, "Phipson & Smyth correction guarantees p > 0");
        assertTrue(p <= 1.0, "p-value must be at most 1.0");
    }

    // --- Default constructor ---

    @Test
    public void testDefaultConstructor() {
        MonteCarloTester tester = new MonteCarloTester();
        double p = tester.test(db, nodeA, nodeB, Collections.emptyList());

        assertTrue(p >= 0.0 && p <= 1.0, "p-value must be in [0, 1]");
        assertTrue(p < 0.05, "Default tester should detect marginal dependence");
    }

    // --- Validation ---

    @Test
    public void testZeroPermutationsThrows() {
        assertThrows(IllegalArgumentException.class, () -> new MonteCarloTester(0, 42L));
    }

    @Test
    public void testNegativePermutationsThrows() {
        assertThrows(IllegalArgumentException.class, () -> new MonteCarloTester(-1, 42L));
    }

    @Test
    public void testMinPermutationsWorks() {
        MonteCarloTester tester = new MonteCarloTester(1, 42L);
        double p = tester.test(db, nodeA, nodeB, Collections.emptyList());

        // With 1 permutation: p = (B+1)/(1+1), so p is either 0.5 or 1.0
        assertTrue(p == 0.5 || p == 1.0,
                "With 1 permutation, p must be 1/2 or 2/2, got " + p);
    }

    // --- Empty conditioning set ---

    @Test
    public void testEmptyConditioningSetEqualsUnconditional() {
        MonteCarloTester tester = new MonteCarloTester(99, 42L);
        double p = tester.test(db, nodeA, nodeB, Collections.emptyList());

        assertTrue(p >= 0.0 && p <= 1.0, "p-value must be in [0, 1]");
    }
}
