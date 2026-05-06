package org.openmarkov.learning.algorithm.pc;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.learning.algorithm.pc.independencetester.CrossEntropyIndependenceTester;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Integration tests for the independence tester used by the PC algorithm.
 *
 * <p>These tests validate the end-to-end statistical
 * behaviour expected from a chi-square / cross-entropy style independence test:
 *
 * <ul>
 *   <li>Marginal dependence should yield a low p-value.</li>
 *   <li>Conditional independence should yield a high p-value.</li>
 * </ul>
 */
public class CrossEntropyIndependenceTesterIntegrationTest {

    private static final double ALPHA = 0.05;

    private static CaseDatabase explainingAwayStyleDatabase;
    private static Node nodeA;
    private static Node nodeB;
    private static Node nodeC;

    @BeforeAll
    static void setUp() {
        Variable varA = new Variable("A", 2);
        Variable varB = new Variable("B", 2);
        Variable varC = new Variable("C", 2);

        List<Variable> variables = List.of(varA, varB, varC);

        List<int[]> rows = new ArrayList<>();

        /*
         * We construct data such that:
         *
         * - A and B are marginally dependent
         * - A and B are conditionally independent given C
         *
         * This corresponds to the common-cause pattern:
         *
         *      C
         *     / \
         *    A   B
         *
         * For each value of C, A and B are sampled independently, but their
         * conditional distributions differ across values of C.
         *
         * C = 0:
         *   P(A=0|C=0)=0.8, P(B=0|C=0)=0.8
         *
         * C = 1:
         *   P(A=0|C=1)=0.2, P(B=0|C=1)=0.2
         *
         * Approximate counts:
         *
         * For C = 0 (100 cases total):
         *   (A,B) = (0,0): 64
         *   (0,1): 16
         *   (1,0): 16
         *   (1,1):  4
         *
         * For C = 1 (100 cases total):
         *   (A,B) = (0,0):  4
         *   (0,1): 16
         *   (1,0): 16
         *   (1,1): 64
         */

        // C = 0
        addCases(rows, new int[]{0, 0, 0}, 64);
        addCases(rows, new int[]{0, 1, 0}, 16);
        addCases(rows, new int[]{1, 0, 0}, 16);
        addCases(rows, new int[]{1, 1, 0}, 4);

        // C = 1
        addCases(rows, new int[]{0, 0, 1}, 4);
        addCases(rows, new int[]{0, 1, 1}, 16);
        addCases(rows, new int[]{1, 0, 1}, 16);
        addCases(rows, new int[]{1, 1, 1}, 64);

        int[][] cases = rows.toArray(new int[0][]);
        explainingAwayStyleDatabase = new CaseDatabase(variables, cases);

        ProbNet probNet = new ProbNet();
        nodeA = new Node(probNet, varA, NodeType.CHANCE);
        nodeB = new Node(probNet, varB, NodeType.CHANCE);
        nodeC = new Node(probNet, varC, NodeType.CHANCE);
    }

    private static void addCases(List<int[]> rows, int[] row, int count) {
        for (int i = 0; i < count; i++) {
            rows.add(row.clone());
        }
    }

    @Test
    @DisplayName("A and B should be marginally dependent")
    void testMarginalDependence() {
        CrossEntropyIndependenceTester tester = new CrossEntropyIndependenceTester();

        double pValue = tester.test(
                explainingAwayStyleDatabase,
                nodeA,
                nodeB,
                Collections.emptyList()
        );

        System.out.printf("Marginal test A ⊥̸ B: p-value = %.8f%n", pValue);

        assertTrue(
                pValue < ALPHA,
                "Expected marginal dependence between A and B, but p-value was " + pValue
        );
    }

    @Test
    @DisplayName("A and B should be conditionally independent given C")
    void testConditionalIndependenceGivenC() {
        CrossEntropyIndependenceTester tester = new CrossEntropyIndependenceTester();

        double pValue = tester.test(
                explainingAwayStyleDatabase,
                nodeA,
                nodeB,
                Collections.singletonList(nodeC)
        );

        System.out.printf("Conditional test A ⊥ B | C: p-value = %.8f%n", pValue);

        assertTrue(
                pValue > ALPHA,
                "Expected conditional independence between A and B given C, but p-value was " + pValue
        );
    }

    @Test
    @DisplayName("Symmetry: test(A, B | C) should match test(B, A | C)")
    void testSymmetryUnderNodeOrder() {
        CrossEntropyIndependenceTester tester = new CrossEntropyIndependenceTester();

        double pValueAB = tester.test(
                explainingAwayStyleDatabase,
                nodeA,
                nodeB,
                Collections.singletonList(nodeC)
        );

        double pValueBA = tester.test(
                explainingAwayStyleDatabase,
                nodeB,
                nodeA,
                Collections.singletonList(nodeC)
        );

        System.out.printf("Symmetry test: p(A,B|C)=%.8f, p(B,A|C)=%.8f%n", pValueAB, pValueBA);

        assertTrue(
                Math.abs(pValueAB - pValueBA) < 1e-12,
                "Expected symmetry in the independence test result."
        );
    }
}