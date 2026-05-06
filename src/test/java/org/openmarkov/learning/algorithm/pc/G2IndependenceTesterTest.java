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
import org.openmarkov.learning.algorithm.pc.independencetester.G2IndependenceTester;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Test suite for {@link G2IndependenceTester}.
 *
 * <p>The tests are organised in two groups:
 *
 * <ol>
 *   <li><b>Standard behaviour</b> — the same synthetic dataset used for
 *       {@link CrossEntropyIndependenceTester} is reused here to verify that
 *       G² with Laplace smoothing still correctly detects marginal dependence
 *       and conditional independence. This ensures the new tester is a valid
 *       drop-in replacement for the existing one on well-populated tables.</li>
 *
 *   <li><b>Sparse-table robustness</b> — a dataset with a rare state is used
 *       to show that Laplace smoothing prevents the statistic from being
 *       distorted when some cells of the contingency table have zero counts.
 *       The same test is run with {@link CrossEntropyIndependenceTester} to
 *       confirm that it gives a misleadingly extreme p-value in that scenario,
 *       while {@link G2IndependenceTester} produces a stable, correct result.</li>
 * </ol>
 */
public class G2IndependenceTesterTest {

    // -------------------------------------------------------------------------
    // Standard dataset: C → A, C → B  (same as CrossEntropyIndependenceTesterTest)
    // -------------------------------------------------------------------------

    private static CaseDatabase standardDb;
    private static Node stdNodeA;
    private static Node stdNodeB;
    private static Node stdNodeC;

    // -------------------------------------------------------------------------
    // Sparse dataset: X → Y with a very rare state of X (P ≈ 0.01)
    // -------------------------------------------------------------------------

    private static CaseDatabase sparseDb;
    private static Node sparseNodeX;
    private static Node sparseNodeY;
    private static Node sparseNodeZ;

    @BeforeAll
    static void setup() {
        buildStandardDatabase();
        buildSparseDatabase();
    }

    /**
     * Standard dataset with the causal structure C → A and C → B.
     * Encodes: A ⊥ B | C, but A and B are marginally dependent.
     */
    private static void buildStandardDatabase() {
        Variable varA = new Variable("A", 2);
        Variable varB = new Variable("B", 2);
        Variable varC = new Variable("C", 2);

        List<Variable> variables = List.of(varA, varB, varC);
        List<int[]> rows = new ArrayList<>();

        // C = 0: P(A=0|C=0)=0.8, P(B=0|C=0)=0.8  → product frequencies
        addCases(rows, new int[]{0, 0, 0}, 64);
        addCases(rows, new int[]{0, 1, 0}, 16);
        addCases(rows, new int[]{1, 0, 0}, 16);
        addCases(rows, new int[]{1, 1, 0}, 4);

        // C = 1: P(A=0|C=1)=0.2, P(B=0|C=1)=0.2  → product frequencies
        addCases(rows, new int[]{0, 0, 1}, 4);
        addCases(rows, new int[]{0, 1, 1}, 16);
        addCases(rows, new int[]{1, 0, 1}, 16);
        addCases(rows, new int[]{1, 1, 1}, 64);

        standardDb = new CaseDatabase(variables, rows.toArray(new int[0][]));

        ProbNet net = new ProbNet();
        stdNodeA = new Node(net, varA, NodeType.CHANCE);
        stdNodeB = new Node(net, varB, NodeType.CHANCE);
        stdNodeC = new Node(net, varC, NodeType.CHANCE);
    }

    /**
     * Sparse dataset designed to expose the zero-cell problem of the
     * unsmoothed chi-square test.
     *
     * <p>Variables are Y (2 states), Z (2 states), X (2 states).
     * The true conditional independence relation is {@code Y ⊥ Z | X}.
     *
     * <p>The X=0 stratum has 900 well-balanced cases (225 per cell), so it
     * contributes nothing to the G² statistic. The X=1 stratum has only 8
     * cases, and by sampling chance they all fall into two diagonal cells:
     *
     * <pre>
     *          Z=0   Z=1
     *   Y=0  [  5     0  ]   X=1 stratum
     *   Y=1  [  0     3  ]
     * </pre>
     *
     * <p>With no cases in two cells, {@link CrossEntropyIndependenceTester}
     * skips those cells and sees perfect Y–Z association in this stratum,
     * which inflates the statistic and leads to a false rejection of
     * independence (p&nbsp;&lt;&nbsp;0.05).
     *
     * <p>{@link G2IndependenceTester} adds one pseudocount to every cell
     * before computing the statistic, which reduces the apparent association
     * in the sparse stratum and produces a p-value above 0.05.
     */
    private static void buildSparseDatabase() {
        Variable varY = new Variable("Y", 2);
        Variable varZ = new Variable("Z", 2);
        Variable varX = new Variable("X", 2);

        // Variable order in CaseDatabase rows: [Y, Z, X]
        List<Variable> variables = List.of(varY, varZ, varX);
        List<int[]> rows = new ArrayList<>();

        // X=0 stratum (900 cases, perfectly balanced → zero G² contribution)
        addCases(rows, new int[]{0, 0, 0}, 225); // Y=0, Z=0, X=0
        addCases(rows, new int[]{0, 1, 0}, 225); // Y=0, Z=1, X=0
        addCases(rows, new int[]{1, 0, 0}, 225); // Y=1, Z=0, X=0
        addCases(rows, new int[]{1, 1, 0}, 225); // Y=1, Z=1, X=0

        // X=1 stratum (8 cases): diagonal pattern due to small sample
        addCases(rows, new int[]{0, 0, 1}, 5);   // Y=0, Z=0, X=1
        // Y=0, Z=1, X=1 → 0 cases (zero cell)
        // Y=1, Z=0, X=1 → 0 cases (zero cell)
        addCases(rows, new int[]{1, 1, 1}, 3);   // Y=1, Z=1, X=1

        sparseDb = new CaseDatabase(variables, rows.toArray(new int[0][]));

        ProbNet net = new ProbNet();
        sparseNodeY = new Node(net, varY, NodeType.CHANCE);
        sparseNodeZ = new Node(net, varZ, NodeType.CHANCE);
        sparseNodeX = new Node(net, varX, NodeType.CHANCE);
    }

    private static void addCases(List<int[]> rows, int[] row, int count) {
        for (int i = 0; i < count; i++) {
            rows.add(row.clone());
        }
    }

    // -------------------------------------------------------------------------
    // Standard-behaviour tests
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("Standard dataset: A and B are marginally dependent (p < 0.05)")
    void testMarginalDependence() {
        G2IndependenceTester tester = new G2IndependenceTester();
        double p = tester.test(standardDb, stdNodeA, stdNodeB, Collections.emptyList());
        System.out.printf("G²  A ⊥̸ B         : p = %.6f%n", p);
        assertTrue(p < 0.05, "Expected marginal dependence (p < 0.05)");
    }

    @Test
    @DisplayName("Standard dataset: A and B are conditionally independent given C (p > 0.05)")
    void testConditionalIndependenceGivenC() {
        G2IndependenceTester tester = new G2IndependenceTester();
        double p = tester.test(standardDb, stdNodeA, stdNodeB, List.of(stdNodeC));
        System.out.printf("G²  A ⊥  B | C    : p = %.6f%n", p);
        assertTrue(p > 0.05, "Expected conditional independence (p > 0.05)");
    }

    // -------------------------------------------------------------------------
    // Sparse-table robustness tests
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("Sparse dataset: G² correctly does not reject Y ⊥ Z | X despite zero cells (p > 0.05)")
    void testSparse_G2_doesNotRejectIndependence() {
        G2IndependenceTester tester = new G2IndependenceTester();
        double p = tester.test(sparseDb, sparseNodeY, sparseNodeZ, List.of(sparseNodeX));
        System.out.printf("G²           Y ⊥ Z | X : p = %.6f%n", p);
        assertTrue(p > 0.05,
                "G² with Laplace smoothing should not reject independence despite sparse cells");
    }

    @Test
    @DisplayName("Sparse dataset: CrossEntropy falsely rejects Y ⊥ Z | X (p < 0.05); G² does not")
    void testSparse_G2RobustWhileCrossEntropyFails() {
        CrossEntropyIndependenceTester crossEntropy = new CrossEntropyIndependenceTester();
        G2IndependenceTester g2 = new G2IndependenceTester();

        double pCE = crossEntropy.test(sparseDb, sparseNodeY, sparseNodeZ, List.of(sparseNodeX));
        double pG2 = g2.test(sparseDb, sparseNodeY, sparseNodeZ, List.of(sparseNodeX));

        System.out.printf("CrossEntropy Y ⊥ Z | X : p = %.6f%n", pCE);
        System.out.printf("G²           Y ⊥ Z | X : p = %.6f%n", pG2);

        // CrossEntropy sees perfect Y–Z association in the X=1 stratum (zero cells
        // are skipped rather than smoothed), which inflates the statistic and
        // leads to a false rejection of the true independence Y ⊥ Z | X.
        assertTrue(pCE < 0.05,
                "CrossEntropyIndependenceTester should falsely reject independence on this sparse table");

        // G² adds one pseudocount to every cell, which dampens the apparent
        // perfect association in the sparse stratum and yields a correct result.
        assertTrue(pG2 > 0.05,
                "G2IndependenceTester should not reject independence thanks to Laplace smoothing");
    }
}
