package org.openmarkov.learning.algorithm.pc;

import org.junit.jupiter.api.BeforeAll;
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
 * Test suite for {@link CrossEntropyIndependenceTester}.
 *
 * <p>This class validates the statistical behaviour of the cross-entropy-based
 * independence tester used by the PC learning algorithm.
 * Rather than testing low-level numerical routines in isolation, these tests
 * exercise the tester in a more integrated way by:
 *
 * <ul>
 *   <li>constructing a synthetic {@link CaseDatabase},</li>
 *   <li>creating the corresponding network variables and nodes,</li>
 *   <li>running the independence test on selected pairs of variables, and</li>
 *   <li>checking whether the returned p-values agree with the expected
 *       probabilistic structure encoded in the synthetic data.</li>
 * </ul>
 *
 * <p>The database used in this class is intentionally designed to represent
 * the following causal pattern:
 *
 * <pre>
 *     C
 *    / \
 *   A   B
 * </pre>
 *
 * <p>That structure induces two key statistical properties:
 *
 * <ul>
 *   <li><b>Marginal dependence</b> between {@code A} and {@code B}, because
 *       both variables are influenced by the common cause {@code C}.</li>
 *   <li><b>Conditional independence</b> between {@code A} and {@code B}
 *       given {@code C}, because once the value of {@code C} is fixed,
 *       the remaining association between {@code A} and {@code B} disappears.</li>
 * </ul>
 *
 * <p>These properties make the dataset particularly suitable for validating
 * the intended semantics of the tester. If the implementation is correct,
 * it should return:
 *
 * <ul>
 *   <li>a <b>small p-value</b> when testing marginal independence between
 *       {@code A} and {@code B}, indicating dependence, and</li>
 *   <li>a <b>large p-value</b> when testing conditional independence
 *       between {@code A} and {@code B} given {@code C}, indicating that
 *       independence cannot be rejected.</li>
 * </ul>
 *
 * <p>These tests serve as functional regression checks for the statistical
 * interpretation of {@link CrossEntropyIndependenceTester#test(CaseDatabase, Node, Node, List)}.
 * In particular, they help detect:
 *
 * <ul>
 *   <li>incorrect computation of conditional cross-entropy,</li>
 *   <li>incorrect conversion from cross-entropy to chi-square statistic,</li>
 *   <li>incorrect computation of degrees of freedom,</li>
 *   <li>confusion between chi-square CDF values and right-tail p-values, and</li>
 *   <li>unexpected changes in the behaviour of the independence tester after
 *       refactoring.</li>
 * </ul>
 *
 * @author Manuel Arias
 */
public class CrossEntropyIndependenceTesterTest {

    /**
     * Synthetic case database used by all tests in this class.
     *
     * <p>Each row represents a complete observation of the variables
     * {@code A}, {@code B}, and {@code C}. The database is built once in
     * {@link #setup()} and then reused by all test methods.
     */
    private static CaseDatabase caseDatabase;

    /**
     * Node representing variable {@code A}.
     *
     * <p>This node is one of the two variables whose marginal and conditional
     * dependence relationships are evaluated in the tests.
     */
    private static Node nodeA;

    /**
     * Node representing variable {@code B}.
     *
     * <p>This node is the second variable whose dependence relations with
     * {@code A} are tested.
     */
    private static Node nodeB;

    /**
     * Node representing variable {@code C}.
     *
     * <p>This node acts as the conditioning variable and is constructed so as
     * to behave as a common cause of {@code A} and {@code B} in the synthetic
     * dataset used in these tests.
     */
    private static Node nodeC;

    /**
     * Builds the synthetic test scenario used by all methods in this class.
     *
     * <p>This initialization method creates:
     *
     * <ul>
     *   <li>three discrete variables {@code A}, {@code B}, and {@code C}, each
     *       with two possible states,</li>
     *   <li>a synthetic {@link CaseDatabase} containing repeated observations
     *       specifically chosen to encode the statistical pattern
     *       {@code A ⫫ B | C} but {@code A} and {@code B} marginally dependent,</li>
     *   <li>a simple {@link ProbNet}, and</li>
     *   <li>three chance nodes associated with the corresponding variables.</li>
     * </ul>
     *
     * <p>The dataset is constructed from approximate conditional distributions:
     *
     * <p>For {@code C = 0}:
     * <pre>
     * P(A=0 | C=0) = 0.8
     * P(B=0 | C=0) = 0.8
     * </pre>
     *
     * <p>For {@code C = 1}:
     * <pre>
     * P(A=0 | C=1) = 0.2
     * P(B=0 | C=1) = 0.2
     * </pre>
     *
     * <p>Within each fixed value of {@code C}, the frequencies are chosen as
     * products of the corresponding conditional probabilities, which induces
     * approximate conditional independence between {@code A} and {@code B}
     * given {@code C}. However, because the conditional distributions differ
     * strongly between {@code C = 0} and {@code C = 1}, the marginal
     * distribution of {@code (A,B)} still reflects dependence.
     *
     * <p>This is a standard and useful synthetic scenario for validating
     * conditional independence tests.
     */
    @BeforeAll
    public static void setup() {
        Variable varA = new Variable("A", 2);
        Variable varB = new Variable("B", 2);
        Variable varC = new Variable("C", 2);

        List<Variable> variables = List.of(varA, varB, varC);

        List<int[]> rows = new ArrayList<>();

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

        caseDatabase = new CaseDatabase(variables, cases);

        ProbNet probNet = new ProbNet();
        nodeA = new Node(probNet, varA, NodeType.CHANCE);
        nodeB = new Node(probNet, varB, NodeType.CHANCE);
        nodeC = new Node(probNet, varC, NodeType.CHANCE);
    }

    /**
     * Adds the same case multiple times to the list of rows used to build the
     * synthetic database.
     *
     * <p>This helper method is a compact way to encode frequency tables as
     * repeated observations. Instead of manually writing dozens of identical
     * rows, a single state configuration is specified together with the number
     * of times it should appear in the final database.
     *
     * <p>The method clones the supplied row before inserting it, ensuring that
     * each stored observation is represented as an independent array and that
     * later modifications to the original argument cannot affect the generated
     * dataset.
     *
     * @param rows  list that accumulates all cases to be included in the
     *              synthetic database
     * @param row   a complete observation encoded as an array of state indices
     *              in the order {@code [A, B, C]}
     * @param count number of times the observation must be inserted
     */
    private static void addCases(List<int[]> rows, int[] row, int count) {
        for (int i = 0; i < count; i++) {
            rows.add(row.clone());
        }
    }

    /**
     * Verifies that {@code A} and {@code B} are detected as marginally dependent.
     *
     * <p>This test calls the independence tester with an empty conditioning set:
     *
     * <pre>
     * A ⫫ B
     * </pre>
     *
     * <p>Because both variables are influenced by the common cause {@code C},
     * they should not be independent in the marginal distribution. Therefore,
     * the null hypothesis of independence is expected to be rejected, which in
     * practice means that the tester should return a sufficiently small p-value.
     *
     * <p>The threshold used here is the conventional significance level
     * {@code 0.05}. Thus, the expected outcome is:
     *
     * <pre>
     * p-value < 0.05
     * </pre>
     *
     * <p>If this test fails, likely causes include:
     *
     * <ul>
     *   <li>incorrect computation of the cross-entropy statistic,</li>
     *   <li>incorrect transformation into a chi-square p-value,</li>
     *   <li>an inverted interpretation of tail probability, or</li>
     *   <li>unexpected changes in the construction of the synthetic dataset.</li>
     * </ul>
     */
    @Test
    public void testMarginalDependence() {
        CrossEntropyIndependenceTester tester = new CrossEntropyIndependenceTester();

        double pValue = tester.test(caseDatabase, nodeA, nodeB, Collections.emptyList());

        System.out.printf("Test A ⊥̸ B (marginal): p-value = %.6f%n", pValue);
        assertTrue(pValue < 0.05, "Expected marginal dependence between A and B.");
    }

    /**
     * Verifies that {@code A} and {@code B} are detected as conditionally
     * independent given {@code C}.
     *
     * <p>This test evaluates the null hypothesis:
     *
     * <pre>
     * A ⫫ B | C
     * </pre>
     *
     * <p>By construction of the synthetic dataset, once the value of
     * {@code C} is fixed, the remaining association between {@code A} and
     * {@code B} disappears. Therefore, the tester is expected to return a
     * sufficiently large p-value, indicating that conditional independence
     * cannot be rejected.
     *
     * <p>Again using the conventional significance level {@code 0.05}, the
     * expected outcome is:
     *
     * <pre>
     * p-value > 0.05
     * </pre>
     *
     * <p>This test is particularly important because it validates the core use
     * case of the independence tester within the PC algorithm: deciding whether
     * an edge between two variables should be removed after conditioning on a
     * candidate separation set.
     *
     * <p>If this test fails, possible explanations include:
     *
     * <ul>
     *   <li>incorrect handling of conditioning sets,</li>
     *   <li>incorrect conditional entropy computation,</li>
     *   <li>incorrect degrees of freedom,</li>
     *   <li>wrong tail semantics in the chi-square probability, or</li>
     *   <li>a synthetic dataset that no longer matches the intended
     *       conditional-independence structure.</li>
     * </ul>
     */
    @Test
    public void testConditionalIndependenceGivenC() {
        CrossEntropyIndependenceTester tester = new CrossEntropyIndependenceTester();

        double pValue = tester.test(caseDatabase, nodeA, nodeB, Collections.singletonList(nodeC));

        System.out.printf("Test A ⊥ B | C: p-value = %.6f%n", pValue);
        assertTrue(pValue > 0.05, "Expected conditional independence between A and B given C.");
    }
}