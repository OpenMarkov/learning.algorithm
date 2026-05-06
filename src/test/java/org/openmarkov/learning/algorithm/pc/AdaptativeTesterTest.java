package org.openmarkov.learning.algorithm.pc;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.learning.algorithm.pc.independencetester.AdaptativeTester;
import org.openmarkov.learning.algorithm.pc.independencetester.CrossEntropyIndependenceTester;
import org.openmarkov.learning.algorithm.pc.independencetester.FisherTester;
import org.openmarkov.learning.algorithm.pc.independencetester.G2IndependenceTester;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Test suite for {@link AdaptativeTester}.
 *
 * <p>The tests are organised in three groups:
 *
 * <ol>
 *   <li><b>Standard behaviour</b> — the same synthetic C → A, C → B dataset
 *       used for the other testers is reused to verify that the adaptive tester
 *       correctly detects marginal dependence and conditional independence.</li>
 *
 *   <li><b>Sparse-table correctness</b> — a dataset with zero cells in one
 *       stratum is used to verify that {@link AdaptativeTester} handles the
 *       case where {@link CrossEntropyIndependenceTester} would fail. On this
 *       2×2 table with small expected counts, the adaptive tester switches to
 *       {@link FisherTester} and correctly does not reject independence.</li>
 *
 *   <li><b>Routing</b> — verifies that the adaptive tester delegates to the
 *       expected sub-tester based on Cochran's criterion:
 *       <ul>
 *         <li>All expected counts ≥ 5 → {@link CrossEntropyIndependenceTester}</li>
 *         <li>Some expected count &lt; 5, 2×2 table → {@link FisherTester}</li>
 *         <li>Some expected count &lt; 5, non-2×2 table → {@link G2IndependenceTester}</li>
 *       </ul>
 *       Routing is verified by checking that the adaptive p-value equals the
 *       sub-tester p-value exactly (since they share the same code path).</li>
 * </ol>
 */
public class AdaptativeTesterTest {

    private static final double ALPHA = 0.05;

    // -------------------------------------------------------------------------
    // Standard dataset: C → A, C → B  (200 cases)
    //
    // Marginal test A ⊥ B (Z=∅):
    //   single stratum, n=200, all expected counts = 50 ≥ 5  → CrossEntropy
    //
    // Conditional test A ⊥ B | C (Z={C}):
    //   C=0 stratum: min expected = 20·20/100 = 4 < 5, table is 2×2 → Fisher
    //   C=1 stratum: same structure
    // -------------------------------------------------------------------------

    private static CaseDatabase standardDb;
    private static Node stdNodeA;
    private static Node stdNodeB;
    private static Node stdNodeC;

    // -------------------------------------------------------------------------
    // Sparse 2×2 dataset: Y ⊥ Z | X with zero cells in the X=1 stratum
    // (same as G2IndependenceTesterTest)
    //
    // X=0 stratum (900 cases): all expected counts = 225 ≥ 5
    // X=1 stratum (8 cases):   min expected = 3·3/8 = 1.125 < 5, table is 2×2 → Fisher
    // -------------------------------------------------------------------------

    private static CaseDatabase sparseDb;
    private static Node sparseNodeY;
    private static Node sparseNodeZ;
    private static Node sparseNodeX;

    // -------------------------------------------------------------------------
    // Sparse non-binary dataset: R ⊥ S | T with zero cells in the T=1 stratum
    // and a 3-state variable R.
    //
    // T=0 stratum (900 cases): perfectly balanced → all expected counts ≥ 5
    // T=1 stratum (9 cases):   min expected < 5, table is 3×2 (non-2×2)  → G2
    // -------------------------------------------------------------------------

    private static CaseDatabase sparseNonBinaryDb;
    private static Node sparseNodeR;
    private static Node sparseNodeS;
    private static Node sparseNodeT;

    @BeforeAll
    static void setUp() {
        buildStandardDatabase();
        buildSparseDatabase();
        buildSparseNonBinaryDatabase();
    }

    private static void buildStandardDatabase() {
        Variable varA = new Variable("A", 2);
        Variable varB = new Variable("B", 2);
        Variable varC = new Variable("C", 2);

        List<int[]> rows = new ArrayList<>();

        // C=0: P(A=0|C=0)=0.8, P(B=0|C=0)=0.8 → product frequencies
        addCases(rows, new int[]{0, 0, 0}, 64);
        addCases(rows, new int[]{0, 1, 0}, 16);
        addCases(rows, new int[]{1, 0, 0}, 16);
        addCases(rows, new int[]{1, 1, 0}, 4);

        // C=1: P(A=0|C=1)=0.2, P(B=0|C=1)=0.2 → product frequencies
        addCases(rows, new int[]{0, 0, 1}, 4);
        addCases(rows, new int[]{0, 1, 1}, 16);
        addCases(rows, new int[]{1, 0, 1}, 16);
        addCases(rows, new int[]{1, 1, 1}, 64);

        standardDb = new CaseDatabase(List.of(varA, varB, varC), rows.toArray(new int[0][]));

        ProbNet net = new ProbNet();
        stdNodeA = new Node(net, varA, NodeType.CHANCE);
        stdNodeB = new Node(net, varB, NodeType.CHANCE);
        stdNodeC = new Node(net, varC, NodeType.CHANCE);
    }

    /**
     * Sparse 2×2 dataset that exposes the zero-cell problem in CrossEntropy.
     *
     * <p>The true relation is Y ⊥ Z | X. The X=1 stratum has only 8 cases
     * concentrated on the diagonal:
     *
     * <pre>
     *          Z=0   Z=1
     *   Y=0  [  5     0  ]   X=1 stratum
     *   Y=1  [  0     3  ]
     * </pre>
     *
     * <p>The minimum expected count in the X=1 stratum is 3·3/8 ≈ 1.125 &lt; 5,
     * so {@link AdaptativeTester} routes to {@link FisherTester}.
     */
    private static void buildSparseDatabase() {
        Variable varY = new Variable("Y", 2);
        Variable varZ = new Variable("Z", 2);
        Variable varX = new Variable("X", 2);

        List<int[]> rows = new ArrayList<>();

        // X=0 stratum (900 cases, perfectly balanced)
        addCases(rows, new int[]{0, 0, 0}, 225);
        addCases(rows, new int[]{0, 1, 0}, 225);
        addCases(rows, new int[]{1, 0, 0}, 225);
        addCases(rows, new int[]{1, 1, 0}, 225);

        // X=1 stratum (8 cases, diagonal → zero cells)
        addCases(rows, new int[]{0, 0, 1}, 5);
        addCases(rows, new int[]{1, 1, 1}, 3);

        sparseDb = new CaseDatabase(List.of(varY, varZ, varX), rows.toArray(new int[0][]));

        ProbNet net = new ProbNet();
        sparseNodeY = new Node(net, varY, NodeType.CHANCE);
        sparseNodeZ = new Node(net, varZ, NodeType.CHANCE);
        sparseNodeX = new Node(net, varX, NodeType.CHANCE);
    }

    /**
     * Sparse non-binary dataset to trigger the G² routing path.
     *
     * <p>R has 3 states, S has 2 states, T has 2 states. The true relation is
     * R ⊥ S | T. The T=0 stratum is well-populated (all expected counts ≥ 5);
     * the T=1 stratum has only 9 cases and zero cells, so some expected counts
     * fall below 5. Since the table is 3×2 (non-2×2), {@link AdaptativeTester}
     * routes to {@link G2IndependenceTester}.
     */
    private static void buildSparseNonBinaryDatabase() {
        Variable varR = new Variable("R", 3);
        Variable varS = new Variable("S", 2);
        Variable varT = new Variable("T", 2);

        List<int[]> rows = new ArrayList<>();

        // T=0 stratum (900 cases, balanced: 150 per cell in the 3×2 table)
        addCases(rows, new int[]{0, 0, 0}, 150);
        addCases(rows, new int[]{0, 1, 0}, 150);
        addCases(rows, new int[]{1, 0, 0}, 150);
        addCases(rows, new int[]{1, 1, 0}, 150);
        addCases(rows, new int[]{2, 0, 0}, 150);
        addCases(rows, new int[]{2, 1, 0}, 150);

        // T=1 stratum (9 cases, concentrated → zero cells and min expected < 5)
        addCases(rows, new int[]{0, 0, 1}, 5);
        addCases(rows, new int[]{1, 1, 1}, 3);
        addCases(rows, new int[]{2, 0, 1}, 1);

        sparseNonBinaryDb = new CaseDatabase(List.of(varR, varS, varT), rows.toArray(new int[0][]));

        ProbNet net = new ProbNet();
        sparseNodeR = new Node(net, varR, NodeType.CHANCE);
        sparseNodeS = new Node(net, varS, NodeType.CHANCE);
        sparseNodeT = new Node(net, varT, NodeType.CHANCE);
    }

    private static void addCases(List<int[]> rows, int[] row, int count) {
        for (int i = 0; i < count; i++) rows.add(row.clone());
    }

    // -------------------------------------------------------------------------
    // Standard-behaviour tests
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("Standard dataset: A and B are marginally dependent (p < 0.05)")
    void testMarginalDependence() {
        AdaptativeTester tester = new AdaptativeTester();
        double p = tester.test(standardDb, stdNodeA, stdNodeB, Collections.emptyList());
        System.out.printf("Adaptive  A ⊥̸ B        : p = %.6f%n", p);
        assertTrue(p < ALPHA, "Expected marginal dependence (p < 0.05), got p = " + p);
    }

    @Test
    @DisplayName("Standard dataset: A and B are conditionally independent given C (p > 0.05)")
    void testConditionalIndependenceGivenC() {
        AdaptativeTester tester = new AdaptativeTester();
        double p = tester.test(standardDb, stdNodeA, stdNodeB, List.of(stdNodeC));
        System.out.printf("Adaptive  A ⊥  B | C   : p = %.6f%n", p);
        assertTrue(p > ALPHA, "Expected conditional independence (p > 0.05), got p = " + p);
    }

    // -------------------------------------------------------------------------
    // Sparse-table correctness tests
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("Sparse 2×2 dataset: AdaptativeTester does not reject Y ⊥ Z | X despite zero cells (p > 0.05)")
    void testSparse_AdaptativeDoesNotRejectIndependence() {
        AdaptativeTester tester = new AdaptativeTester();
        double p = tester.test(sparseDb, sparseNodeY, sparseNodeZ, List.of(sparseNodeX));
        System.out.printf("Adaptive  Y ⊥  Z | X   : p = %.6f%n", p);
        assertTrue(p > ALPHA,
                "AdaptativeTester should not reject independence on sparse 2×2 data, got p = " + p);
    }

    @Test
    @DisplayName("Sparse 2×2 dataset: CrossEntropy falsely rejects Y ⊥ Z | X; AdaptativeTester does not")
    void testSparse_AdaptativeSucceedsWhereCrossEntropyFails() {
        CrossEntropyIndependenceTester crossEntropy = new CrossEntropyIndependenceTester();
        AdaptativeTester adaptive = new AdaptativeTester();

        double pCE = crossEntropy.test(sparseDb, sparseNodeY, sparseNodeZ, List.of(sparseNodeX));
        double pAd = adaptive.test(sparseDb, sparseNodeY, sparseNodeZ, List.of(sparseNodeX));

        System.out.printf("CrossEntropy  Y ⊥ Z | X : p = %.6f%n", pCE);
        System.out.printf("Adaptive      Y ⊥ Z | X : p = %.6f%n", pAd);

        assertTrue(pCE < ALPHA,
                "CrossEntropyIndependenceTester should falsely reject independence on sparse 2×2 data");
        assertTrue(pAd > ALPHA,
                "AdaptativeTester should not reject independence on sparse 2×2 data");
    }

    // -------------------------------------------------------------------------
    // Routing tests
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("Routing: all expected counts ≥ 5 → delegates to CrossEntropyIndependenceTester")
    void testRouting_DelegatesToCrossEntropyWhenExpectedCountsSufficient() {
        // Marginal test A ⊥ B (Z=∅): single stratum with n=200,
        // all expected counts = 50 ≥ 5 → CrossEntropy path.
        CrossEntropyIndependenceTester crossEntropy = new CrossEntropyIndependenceTester();
        AdaptativeTester adaptive = new AdaptativeTester();

        double pCE = crossEntropy.test(standardDb, stdNodeA, stdNodeB, Collections.emptyList());
        double pAd = adaptive.test(standardDb, stdNodeA, stdNodeB, Collections.emptyList());

        System.out.printf("CrossEntropy  A ⊥̸ B       : p = %.8f%n", pCE);
        System.out.printf("Adaptive      A ⊥̸ B       : p = %.8f%n", pAd);

        assertEquals(pCE, pAd, 1e-12,
                "AdaptativeTester should produce the exact same p-value as CrossEntropy when all expected counts ≥ 5");
    }

    @Test
    @DisplayName("Routing: sparse 2×2 table → delegates to FisherTester")
    void testRouting_DelegatesToFisherOnSparse2x2Table() {
        // Sparse 2×2 dataset: X=1 stratum has min expected ≈ 1.125 < 5
        // and both Y and Z are binary → Fisher path.
        FisherTester fisher = new FisherTester();
        AdaptativeTester adaptive = new AdaptativeTester();

        double pFi = fisher.test(sparseDb, sparseNodeY, sparseNodeZ, List.of(sparseNodeX));
        double pAd = adaptive.test(sparseDb, sparseNodeY, sparseNodeZ, List.of(sparseNodeX));

        System.out.printf("Fisher    Y ⊥ Z | X : p = %.8f%n", pFi);
        System.out.printf("Adaptive  Y ⊥ Z | X : p = %.8f%n", pAd);

        assertEquals(pFi, pAd, 1e-12,
                "AdaptativeTester should produce the exact same p-value as FisherTester for sparse 2×2 tables");
    }

    @Test
    @DisplayName("Routing: sparse non-binary table → delegates to G2IndependenceTester")
    void testRouting_DelegatesToG2OnSparseNonBinaryTable() {
        // Sparse 3×2 dataset: T=1 stratum has min expected < 5
        // and R has 3 states → G2 path.
        G2IndependenceTester g2 = new G2IndependenceTester();
        AdaptativeTester adaptive = new AdaptativeTester();

        double pG2 = g2.test(sparseNonBinaryDb, sparseNodeR, sparseNodeS, List.of(sparseNodeT));
        double pAd = adaptive.test(sparseNonBinaryDb, sparseNodeR, sparseNodeS, List.of(sparseNodeT));

        System.out.printf("G2        R ⊥ S | T : p = %.8f%n", pG2);
        System.out.printf("Adaptive  R ⊥ S | T : p = %.8f%n", pAd);

        assertEquals(pG2, pAd, 1e-12,
                "AdaptativeTester should produce the exact same p-value as G2IndependenceTester for sparse non-binary tables");
    }
}
