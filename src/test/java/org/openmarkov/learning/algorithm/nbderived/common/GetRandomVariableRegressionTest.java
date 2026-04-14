package org.openmarkov.learning.algorithm.nbderived.common;

import org.junit.jupiter.api.Test;
import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.potential.TablePotential;
import org.openmarkov.learning.algorithm.nbderived.treeaugmentednb.TreeAugmentedNBAlgorithm;
import org.openmarkov.learning.metric.Metric;

import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression test for bug #8: off-by-one in getRandomVariable().
 *
 * <p>The original code used {@code nextInt(size - 1)}, which excluded the last
 * element from the list of non-root variables. The fix changed this to
 * {@code nextInt(size)}.
 *
 * <p>These tests verify:
 * <ul>
 *   <li>With exactly 1 non-root variable, the method returns it (the old code
 *       would call {@code nextInt(0)} and throw {@code IllegalArgumentException})</li>
 *   <li>With multiple non-root variables, all are reachable over many calls</li>
 *   <li>The root (class) variable is never returned</li>
 * </ul>
 *
 * @author Manuel Arias
 */
public class GetRandomVariableRegressionTest {

    /**
     * With exactly 1 non-root variable, getRandomVariable must return it.
     * The old code (nextInt(size-1) = nextInt(0)) would throw IllegalArgumentException.
     */
    @Test
    public void testSingleNonRootVariable_ReturnsIt() {
        ProbNet probNet = new ProbNet();
        Variable classVar = new Variable("Class", 2);
        Variable f1 = new Variable("F1", 2);
        probNet.addNode(classVar, NodeType.CHANCE);
        probNet.addNode(f1, NodeType.CHANCE);

        TestableTAN alg = new TestableTAN(probNet);
        alg.setClassVariableName("Class");

        Variable result = alg.callGetRandomVariable();
        assertEquals("F1", result.getName(),
                "Single non-root variable should always be returned");
    }

    /**
     * With multiple non-root variables, all should be reachable.
     * The old code would never return the last element in the list.
     */
    @Test
    public void testAllNonRootVariablesReachable() {
        ProbNet probNet = new ProbNet();
        Variable classVar = new Variable("Class", 2);
        Variable f1 = new Variable("F1", 2);
        Variable f2 = new Variable("F2", 2);
        Variable f3 = new Variable("F3", 2);
        probNet.addNode(classVar, NodeType.CHANCE);
        probNet.addNode(f1, NodeType.CHANCE);
        probNet.addNode(f2, NodeType.CHANCE);
        probNet.addNode(f3, NodeType.CHANCE);

        TestableTAN alg = new TestableTAN(probNet);
        alg.setClassVariableName("Class");

        Set<String> observed = new HashSet<>();
        // With 3 features, run enough iterations to very likely see all
        for (int i = 0; i < 200; i++) {
            Variable v = alg.callGetRandomVariable();
            observed.add(v.getName());
        }

        assertTrue(observed.contains("F1"), "F1 should be reachable");
        assertTrue(observed.contains("F2"), "F2 should be reachable");
        assertTrue(observed.contains("F3"), "F3 should be reachable");
        assertFalse(observed.contains("Class"), "Class variable should never be returned");
    }

    /**
     * getRandomVariable should never return the class (root) variable.
     */
    @Test
    public void testNeverReturnsRootVariable() {
        ProbNet probNet = new ProbNet();
        Variable classVar = new Variable("Class", 2);
        Variable f1 = new Variable("F1", 2);
        probNet.addNode(classVar, NodeType.CHANCE);
        probNet.addNode(f1, NodeType.CHANCE);

        TestableTAN alg = new TestableTAN(probNet);
        alg.setClassVariableName("Class");

        for (int i = 0; i < 50; i++) {
            Variable v = alg.callGetRandomVariable();
            assertNotEquals("Class", v.getName(),
                    "getRandomVariable should never return the class variable");
        }
    }

    // --- Testable subclass to expose the protected method ---

    private static class TestableTAN extends TreeAugmentedNBAlgorithm {

        TestableTAN(ProbNet probNet) {
            super(probNet, null, new NoOpMetric(), 1.0);
        }

        Variable callGetRandomVariable() {
            return getRandomVariable();
        }
    }

    private static class NoOpMetric extends Metric {
        @Override
        public double getScore(PNEdit edit) {
            return 0;
        }

        @Override
        public double score(TablePotential nodePotential) {
            return 0;
        }
    }
}
