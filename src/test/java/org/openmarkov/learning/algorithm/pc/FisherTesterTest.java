package org.openmarkov.learning.algorithm.pc;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.learning.algorithm.pc.independencetester.FisherTester;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link FisherTester}.
 * <p>
 * Validates both the exact Fisher path (binary variables) and the
 * Pearson chi-square fallback (variables with more than two states).
 * Uses the same synthetic database pattern as {@link CrossEntropyIndependenceTesterTest}:
 * C is a common cause of A and B, so A and B are marginally dependent
 * but conditionally independent given C.
 *
 * @author Manuel Arias
 */
public class FisherTesterTest {

    // --- Binary scenario (exact Fisher path) ---

    private static CaseDatabase binaryDb;
    private static Node bNodeA, bNodeB, bNodeC;

    // --- Ternary scenario (chi-square fallback path) ---

    private static CaseDatabase ternaryDb;
    private static Node tNodeA, tNodeB, tNodeC;

    @BeforeAll
    public static void setup() {
        setupBinary();
        setupTernary();
    }

    /**
     * Binary variables (2 states each). This exercises the exact Fisher code path.
     * Structure: C -> A, C -> B. Frequencies chosen so that A _|_ B | C.
     */
    private static void setupBinary() {
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

        binaryDb = new CaseDatabase(variables, rows.toArray(new int[0][]));

        ProbNet net = new ProbNet();
        bNodeA = new Node(net, varA, NodeType.CHANCE);
        bNodeB = new Node(net, varB, NodeType.CHANCE);
        bNodeC = new Node(net, varC, NodeType.CHANCE);
    }

    /**
     * Ternary variables (3 states each). This exercises the Pearson chi-square fallback.
     * Same logical structure: C -> A, C -> B with conditional independence.
     */
    private static void setupTernary() {
        Variable varA = new Variable("A", 3);
        Variable varB = new Variable("B", 3);
        Variable varC = new Variable("C", 2);
        List<Variable> variables = List.of(varA, varB, varC);

        List<int[]> rows = new ArrayList<>();
        // C=0: A and B both concentrate on state 0
        // P(A|C=0) ~ (0.7, 0.2, 0.1), P(B|C=0) ~ (0.7, 0.2, 0.1)
        // Product frequencies for 100 cases at C=0:
        addCases(rows, new int[]{0, 0, 0}, 49);
        addCases(rows, new int[]{0, 1, 0}, 14);
        addCases(rows, new int[]{0, 2, 0}, 7);
        addCases(rows, new int[]{1, 0, 0}, 14);
        addCases(rows, new int[]{1, 1, 0}, 4);
        addCases(rows, new int[]{1, 2, 0}, 2);
        addCases(rows, new int[]{2, 0, 0}, 7);
        addCases(rows, new int[]{2, 1, 0}, 2);
        addCases(rows, new int[]{2, 2, 0}, 1);

        // C=1: A and B both concentrate on state 2
        // P(A|C=1) ~ (0.1, 0.2, 0.7), P(B|C=1) ~ (0.1, 0.2, 0.7)
        addCases(rows, new int[]{0, 0, 1}, 1);
        addCases(rows, new int[]{0, 1, 1}, 2);
        addCases(rows, new int[]{0, 2, 1}, 7);
        addCases(rows, new int[]{1, 0, 1}, 2);
        addCases(rows, new int[]{1, 1, 1}, 4);
        addCases(rows, new int[]{1, 2, 1}, 14);
        addCases(rows, new int[]{2, 0, 1}, 7);
        addCases(rows, new int[]{2, 1, 1}, 14);
        addCases(rows, new int[]{2, 2, 1}, 49);

        ternaryDb = new CaseDatabase(variables, rows.toArray(new int[0][]));

        ProbNet net = new ProbNet();
        tNodeA = new Node(net, varA, NodeType.CHANCE);
        tNodeB = new Node(net, varB, NodeType.CHANCE);
        tNodeC = new Node(net, varC, NodeType.CHANCE);
    }

    private static void addCases(List<int[]> rows, int[] row, int count) {
        for (int i = 0; i < count; i++) {
            rows.add(row.clone());
        }
    }

    // --- Exact Fisher tests (binary) ---

    @Test
    public void testBinaryMarginalDependence() {
        FisherTester tester = new FisherTester();
        double pValue = tester.test(binaryDb, bNodeA, bNodeB, Collections.emptyList());

        System.out.printf("Fisher  A ⊥̸ B         : p = %.8f%n", pValue);
        assertTrue(pValue < 0.05, "Expected marginal dependence between binary A and B.");
    }

    @Test
    public void testBinaryConditionalIndependence() {
        FisherTester tester = new FisherTester();
        double pValue = tester.test(binaryDb, bNodeA, bNodeB, Collections.singletonList(bNodeC));

        System.out.printf("Fisher  A ⊥  B | C    : p = %.8f%n", pValue);
        assertTrue(pValue > 0.05, "Expected conditional independence between binary A and B given C.");
    }

    // --- Chi-square fallback tests (ternary) ---

    @Test
    public void testTernaryMarginalDependence() {
        FisherTester tester = new FisherTester();
        double pValue = tester.test(ternaryDb, tNodeA, tNodeB, Collections.emptyList());

        System.out.printf("Fisher(χ²) A ⊥̸ B      : p = %.8f%n", pValue);
        assertTrue(pValue < 0.05, "Expected marginal dependence between ternary A and B.");
    }

    @Test
    public void testTernaryConditionalIndependence() {
        FisherTester tester = new FisherTester();
        double pValue = tester.test(ternaryDb, tNodeA, tNodeB, Collections.singletonList(tNodeC));

        System.out.printf("Fisher(χ²) A ⊥  B | C : p = %.8f%n", pValue);
        assertTrue(pValue > 0.05, "Expected conditional independence between ternary A and B given C.");
    }
}
