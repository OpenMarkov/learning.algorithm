package org.openmarkov.learning.algorithm.nbderived.common;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.potential.TablePotential;
import org.openmarkov.learning.metric.Metric;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link MaximumWeightSpanningTree}.
 *
 * @author Manuel Arias
 */
public class MaximumWeightSpanningTreeTest {

    private ProbNet probNet;
    private Variable vA, vB, vC, vD;
    private Node nA, nB, nC, nD;

    @BeforeEach
    void setup() {
        probNet = new ProbNet();
        vA = new Variable("A");
        vB = new Variable("B");
        vC = new Variable("C");
        vD = new Variable("D");
        nA = probNet.addNode(vA, NodeType.CHANCE);
        nB = probNet.addNode(vB, NodeType.CHANCE);
        nC = probNet.addNode(vC, NodeType.CHANCE);
        nD = probNet.addNode(vD, NodeType.CHANCE);
    }

    @Test
    void testIsBuiltInitiallyFalse() {
        MaximumWeightSpanningTree mwst = new MaximumWeightSpanningTree();
        assertFalse(mwst.isBuilt());
    }

    @Test
    void testBuildCreatesSpanningTree() {
        // Scores: A-B=10, A-C=5, A-D=1, B-C=8, B-D=3, C-D=2
        // Kruskal picks: A-B(10), B-C(8), B-D(3) — 3 edges for 4 nodes
        FixedScoreMetric metric = new FixedScoreMetric();
        metric.setScore("A", "B", 10);
        metric.setScore("A", "C", 5);
        metric.setScore("A", "D", 1);
        metric.setScore("B", "C", 8);
        metric.setScore("B", "D", 3);
        metric.setScore("C", "D", 2);

        MaximumWeightSpanningTree mwst = new MaximumWeightSpanningTree();
        mwst.build(probNet, metric, List.of(nA, nB, nC, nD));

        assertTrue(mwst.isBuilt());
        assertEquals(3, mwst.getUndirectedEdges().size(),
                "Spanning tree of 4 nodes should have 3 edges");
    }

    @Test
    void testRedirectFromRoot() {
        FixedScoreMetric metric = new FixedScoreMetric();
        // Linear tree: A-B-C-D (A-B=10, B-C=8, C-D=6, others=1)
        metric.setScore("A", "B", 10);
        metric.setScore("B", "C", 8);
        metric.setScore("C", "D", 6);
        metric.setScore("A", "C", 1);
        metric.setScore("A", "D", 1);
        metric.setScore("B", "D", 1);

        MaximumWeightSpanningTree mwst = new MaximumWeightSpanningTree();
        mwst.build(probNet, metric, List.of(nA, nB, nC, nD));
        mwst.redirect(vA, probNet);

        assertEquals(3, mwst.getDirectedEdges().size());
        // All edges should point away from A
        assertTrue(mwst.contains(vA, vB), "Should have A->B");
        assertTrue(mwst.contains(vB, vC), "Should have B->C");
        assertTrue(mwst.contains(vC, vD), "Should have C->D");
        // Reverse should not be present
        assertFalse(mwst.contains(vB, vA));
        assertFalse(mwst.contains(vC, vB));
        assertFalse(mwst.contains(vD, vC));
    }

    @Test
    void testRedirectFromMiddle() {
        FixedScoreMetric metric = new FixedScoreMetric();
        // Linear tree: A-B-C (A-B=10, B-C=8, A-C=1)
        metric.setScore("A", "B", 10);
        metric.setScore("B", "C", 8);
        metric.setScore("A", "C", 1);

        MaximumWeightSpanningTree mwst = new MaximumWeightSpanningTree();
        mwst.build(probNet, metric, List.of(nA, nB, nC));
        mwst.redirect(vB, probNet);

        assertTrue(mwst.contains(vB, vA), "Should have B->A");
        assertTrue(mwst.contains(vB, vC), "Should have B->C");
        assertFalse(mwst.contains(vA, vB));
        assertFalse(mwst.contains(vC, vB));
    }

    @Test
    void testContainsReturnsFalseBeforeRedirect() {
        FixedScoreMetric metric = new FixedScoreMetric();
        metric.setScore("A", "B", 10);
        metric.setScore("B", "C", 8);
        metric.setScore("A", "C", 1);

        MaximumWeightSpanningTree mwst = new MaximumWeightSpanningTree();
        mwst.build(probNet, metric, List.of(nA, nB, nC));

        // Before redirect, directed edges are empty
        assertFalse(mwst.contains(vA, vB));
    }

    @Test
    void testKruskalSkipsCycleEdges() {
        // Complete graph K3: A-B=10, B-C=8, A-C=5
        // Kruskal picks A-B(10), B-C(8) — skips A-C because it would form a cycle
        FixedScoreMetric metric = new FixedScoreMetric();
        metric.setScore("A", "B", 10);
        metric.setScore("B", "C", 8);
        metric.setScore("A", "C", 5);

        MaximumWeightSpanningTree mwst = new MaximumWeightSpanningTree();
        mwst.build(probNet, metric, List.of(nA, nB, nC));

        assertEquals(2, mwst.getUndirectedEdges().size(),
                "Spanning tree of 3 nodes should have 2 edges");
    }

    /**
     * Stub metric that returns configured scores for variable pairs.
     */
    private static class FixedScoreMetric extends Metric {
        private final java.util.Map<String, Double> scores = new java.util.HashMap<>();

        void setScore(String v1, String v2, double score) {
            String key = v1.compareTo(v2) < 0 ? v1 + "-" + v2 : v2 + "-" + v1;
            scores.put(key, score);
        }

        @Override
        public double getScore(PNEdit edit) {
            if (edit instanceof AddLinkEdit addLink) {
                String n1 = addLink.getVariableFrom().getName();
                String n2 = addLink.getVariableTo().getName();
                String key = n1.compareTo(n2) < 0 ? n1 + "-" + n2 : n2 + "-" + n1;
                return scores.getOrDefault(key, 0.0);
            }
            return 0.0;
        }

        @Override
        public double score(TablePotential nodePotential) {
            return 0;
        }
    }
}
