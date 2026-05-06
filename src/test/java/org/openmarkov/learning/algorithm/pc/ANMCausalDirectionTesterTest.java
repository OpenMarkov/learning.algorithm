package org.openmarkov.learning.algorithm.pc;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.learning.algorithm.pc.independencetester.ANMCausalDirectionTester;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link ANMCausalDirectionTester}.
 *
 * <p>The Additive Noise Model (ANM) test fits a modal regression f(x)
 * and checks whether the residual noise is independent of X. A high
 * p-value supports the X -> Y direction.
 *
 * <p>The primary dataset uses an asymmetric structure:
 * X (binary) -> Y (ternary) where Y = shift(X) + uniform noise.
 * The forward noise is independent of X (high p-value), but the
 * reverse regression produces noise that depends on Y (low p-value).
 *
 * @author Manuel Arias
 */
public class ANMCausalDirectionTesterTest {

    // --- Asymmetric dataset: X (binary) -> Y (ternary) ---

    private static CaseDatabase asymDb;
    private static Node nodeX, nodeY;

    // --- Additional ProbNet for variable-not-found test ---

    private static Node nodeNotInDb;

    @BeforeAll
    static void setup() {
        buildAsymmetricDataset();
    }

    /**
     * Asymmetric dataset: X binary (0, 1), Y ternary (0, 1, 2).
     *
     * <p>Causal mechanism: Y = X + noise (mod 3), with noise distribution
     * P(noise=0) = 0.8, P(noise=1) = 0.1, P(noise=2) = 0.1, independent of X.
     *
     * <ul>
     *   <li>X=0: Y ~ (0.8, 0.1, 0.1) -> f(0) = 0</li>
     *   <li>X=1: Y ~ (0.1, 0.8, 0.1) -> f(1) = 1</li>
     * </ul>
     *
     * <p>Forward direction (X->Y): noise eps = (Y - f(X) + 3) % 3
     * has the same distribution {0.8, 0.1, 0.1} for both X values
     * -> eps independent of X -> high p-value.
     *
     * <p>Reverse direction (Y->X): the noise distribution varies across Y
     * -> eps depends on Y -> low p-value.
     */
    private static void buildAsymmetricDataset() {
        Variable varX = new Variable("X", 2);
        Variable varY = new Variable("Y", 3);
        List<Variable> variables = List.of(varX, varY);

        List<int[]> rows = new ArrayList<>();
        // X=0 (200 cases): Y ~ (0.8, 0.1, 0.1)
        addCases(rows, new int[]{0, 0}, 160);
        addCases(rows, new int[]{0, 1}, 20);
        addCases(rows, new int[]{0, 2}, 20);
        // X=1 (200 cases): Y ~ (0.1, 0.8, 0.1)
        addCases(rows, new int[]{1, 0}, 20);
        addCases(rows, new int[]{1, 1}, 160);
        addCases(rows, new int[]{1, 2}, 20);

        asymDb = new CaseDatabase(variables, rows.toArray(new int[0][]));

        ProbNet net = new ProbNet();
        nodeX = new Node(net, varX, NodeType.CHANCE);
        nodeY = new Node(net, varY, NodeType.CHANCE);

        // Node with a variable not present in the database
        ProbNet otherNet = new ProbNet();
        Variable varZ = new Variable("Z", 2);
        nodeNotInDb = new Node(otherNet, varZ, NodeType.CHANCE);
    }

    private static void addCases(List<int[]> rows, int[] row, int count) {
        for (int i = 0; i < count; i++) {
            rows.add(row.clone());
        }
    }

    // --- Forward direction tests ---

    @Test
    public void testForwardDirection_HighPValue() {
        ANMCausalDirectionTester tester = new ANMCausalDirectionTester();
        double p = tester.testDirection(asymDb, nodeX, nodeY);

        System.out.printf("ANM  X -> Y : p = %.6f%n", p);
        assertTrue(p > 0.05,
                "Forward direction (X->Y) should have high p-value (noise independent of X), got p = " + p);
    }

    @Test
    public void testReverseDirection_LowPValue() {
        ANMCausalDirectionTester tester = new ANMCausalDirectionTester();
        double p = tester.testDirection(asymDb, nodeY, nodeX);

        System.out.printf("ANM  Y -> X : p = %.6f%n", p);
        assertTrue(p < 0.05,
                "Reverse direction (Y->X) should have low p-value (noise depends on Y), got p = " + p);
    }

    @Test
    public void testForwardBetterThanReverse() {
        ANMCausalDirectionTester tester = new ANMCausalDirectionTester();
        double pForward = tester.testDirection(asymDb, nodeX, nodeY);
        double pReverse = tester.testDirection(asymDb, nodeY, nodeX);

        System.out.printf("ANM  X -> Y : p = %.6f   Y -> X : p = %.6f%n", pForward, pReverse);
        assertTrue(pForward > pReverse,
                "Forward p-value should be higher than reverse p-value");
    }

    // --- Variable not found ---

    @Test
    public void testVariableNotInDatabase_ReturnsZero() {
        ANMCausalDirectionTester tester = new ANMCausalDirectionTester();
        double p = tester.testDirection(asymDb, nodeNotInDb, nodeY);

        assertEquals(0.0, p, "Should return 0 when a variable is not in the database");
    }

    @Test
    public void testSecondVariableNotInDatabase_ReturnsZero() {
        ANMCausalDirectionTester tester = new ANMCausalDirectionTester();
        double p = tester.testDirection(asymDb, nodeX, nodeNotInDb);

        assertEquals(0.0, p, "Should return 0 when the second variable is not in the database");
    }

    // --- p-value range ---

    @Test
    public void testPValueInValidRange() {
        ANMCausalDirectionTester tester = new ANMCausalDirectionTester();
        double p = tester.testDirection(asymDb, nodeX, nodeY);

        assertTrue(p >= 0.0 && p <= 1.0, "p-value must be in [0, 1], got " + p);
    }

    // --- Binary variables ---

    @Test
    public void testBinaryVariables() {
        Variable varA = new Variable("A", 2);
        Variable varB = new Variable("B", 2);
        List<Variable> variables = List.of(varA, varB);

        // A -> B with P(B=A) = 0.9
        List<int[]> rows = new ArrayList<>();
        addCases(rows, new int[]{0, 0}, 90);
        addCases(rows, new int[]{0, 1}, 10);
        addCases(rows, new int[]{1, 0}, 10);
        addCases(rows, new int[]{1, 1}, 90);

        CaseDatabase binaryDb = new CaseDatabase(variables, rows.toArray(new int[0][]));
        ProbNet net = new ProbNet();
        Node nA = new Node(net, varA, NodeType.CHANCE);
        Node nB = new Node(net, varB, NodeType.CHANCE);

        ANMCausalDirectionTester tester = new ANMCausalDirectionTester();
        double p = tester.testDirection(binaryDb, nA, nB);

        System.out.printf("ANM binary A -> B : p = %.6f%n", p);
        assertTrue(p >= 0.0 && p <= 1.0, "p-value must be in [0, 1]");
        // For a symmetric binary relationship, both directions yield similar noise
        // so we just check it runs and returns a valid p-value
    }
}
