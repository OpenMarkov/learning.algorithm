package org.openmarkov.learning.algorithm.pc;

import org.junit.jupiter.api.Test;
import org.openmarkov.core.model.graph.Link;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.learning.algorithm.pc.independencetester.IndependenceTester;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class PCAlgorithmTest {

    // Simple stub for IndependenceTester
    static class MockIndependenceTester implements IndependenceTester {
        private final Map<String, Double> pValues = new HashMap<>();

        public void setIndependence(String node1, String node2, double pValue) {
            // Sort names to ensure consistency
            String key = getKey(node1, node2);
            pValues.put(key, pValue);
        }

        private String getKey(String n1, String n2) {
            return (n1.compareTo(n2) < 0) ? n1 + "-" + n2 : n2 + "-" + n1;
        }

        @Override
        public double test(CaseDatabase caseDatabase, Node node1, Node node2, List<Node> adjacencySubset) {
            String key = getKey(node1.getName(), node2.getName());
            // Default to dependent (low p-value) if not specified
            return pValues.getOrDefault(key, 0.0);
        }
    }

    @Test
    public void testSkeletonDiscovery() throws Exception {
        // Setup: A -> B <- C (V-structure)
        // Correct skeleton: A-B, B-C. A and C are independent given empty set.

        ProbNet probNet = new ProbNet();
        Variable varA = new Variable("A");
        Variable varB = new Variable("B");
        Variable varC = new Variable("C");
        Node nodeA = probNet.addNode(varA, NodeType.CHANCE);
        Node nodeB = probNet.addNode(varB, NodeType.CHANCE);
        Node nodeC = probNet.addNode(varC, NodeType.CHANCE);

        // Start with complete graph (lines 49-54 of
        // IndependenceRelationsAlgorithm.init)
        // But PCAlgorithm constructor just takes the net.
        // We need to simulate the initial state where it might have all links or we
        // rely on 'init'
        // The algorithm itself usually starts by assuming complete graph if init is
        // called properly.
        // However, PCAlgorithm uses 'current' structure of probNet.
        // Let's manually add links A-B, B-C, A-C to simulate starting point or allow
        // logic to remove them.
        probNet.addLink(nodeA, nodeB, false);
        probNet.addLink(nodeB, nodeC, false);
        probNet.addLink(nodeA, nodeC, false);

        MockIndependenceTester tester = new MockIndependenceTester();
        // A _|_ C given empty set (p-value > alpha)
        // Let alpha = 0.05. We set p-value = 0.5 for A-C
        tester.setIndependence("A", "C", 0.5);
        // Others are dependent (p-value = 0.0)

        PCAlgorithm pc = new PCAlgorithm(probNet, null, 0.05, tester, 0.05, null);

        // Run logic. Since PCAlgorithm is step-based (getBestEdit), we might need to
        // loop.
        // But typically one would use the 'init' method first.
        // For this unit test, we can call 'getOptimalEdit' repeatedly.

        // We expect A-C to be removed.
        boolean linkRemoved = false;
        int maxSteps = 100;
        for (int i = 0; i < maxSteps; i++) {
            var proposal = pc.getOptimalEdit(false, true); // onlyAllowed=false, onlyPositive=true
            if (proposal == null)
                break;

            // Apply edit
            proposal.getEdit().executeEdit();

            if (proposal.getEdit() instanceof org.openmarkov.core.action.base.linkEdits.RemoveLinkEdit) {
                org.openmarkov.core.action.base.linkEdits.RemoveLinkEdit edit = (org.openmarkov.core.action.base.linkEdits.RemoveLinkEdit) proposal
                        .getEdit();
                if (edit.toString().contains("A") && edit.toString().contains("C")) {
                    linkRemoved = true;
                }
            }
        }

        assertTrue(linkRemoved, "Link A-C should be removed based on independence test");
        assertNull(probNet.getLink(nodeA, nodeC, false), "Link A-C should not exist in ProbNet");

        // Links might be oriented as A->B and C->B (collider)
        boolean abExists = probNet.getLink(nodeA, nodeB, false) != null || probNet.getLink(nodeA, nodeB, true) != null;
        assertTrue(abExists, "Link A-B should still exist");

        boolean bcExists = probNet.getLink(nodeB, nodeC, false) != null || probNet.getLink(nodeB, nodeC, true) != null
                || probNet.getLink(nodeC, nodeB, true) != null;
        assertTrue(bcExists, "Link B-C should still exist");
    }

    @Test
    public void testColliderOrientation() throws Exception {
        // Setup: A - B - C. A removed-linked-to C.
        // B is NOT in separation set of {A, C}.
        // Result should be A -> B <- C

        ProbNet probNet = new ProbNet();
        Variable varA = new Variable("A");
        Variable varB = new Variable("B");
        Variable varC = new Variable("C");
        Node nodeA = probNet.addNode(varA, NodeType.CHANCE);
        Node nodeB = probNet.addNode(varB, NodeType.CHANCE);
        Node nodeC = probNet.addNode(varC, NodeType.CHANCE);

        // Start with skeleton A-B-C
        probNet.addLink(nodeA, nodeB, false);
        probNet.addLink(nodeB, nodeC, false);

        MockIndependenceTester tester = new MockIndependenceTester();
        // Assume A _|_ C was found with SepSet = {} (B not included)
        // checking the internal cache might be tricky unless we populate it or simulate
        // the removal
        // PCAlgorithm.cache is protected! We can subclass or access it if in same
        // package.
        // Package is org.openmarkov.learning.algorithm.pc, so we ARE in same package.

        PCAlgorithm pc = new PCAlgorithm(probNet, null, 0.05, tester, 0.05, null);

        // Manually populate cache to simulate that A-C were removed with empty sepset
        org.openmarkov.learning.algorithm.pc.util.NodePair pair = new org.openmarkov.learning.algorithm.pc.util.NodePair(
                nodeA, nodeC);
        pc.cache.put(pair,
                new PCEditMotivation(0.5, java.util.Collections.emptyList()));

        // We force phase to HEAD_TO_HEAD_ORIENTATION via reflection or by exhausting
        // previous phase?
        // PCAlgorithm.phase is private. But 'getOrientationEdit' is public.
        // 'transitionToNextPhase' is private.
        // Best way: run until it orients.

        boolean oriented = false;
        int maxSteps = 20;
        for (int i = 0; i < maxSteps; i++) {
            // We use getOrientationEdit directly to test this specific logic
            var proposal = pc.getOrientationEdit(false);
            if (proposal != null) {
                proposal.getEdit().executeEdit();
                oriented = true;
            } else {
                break;
            }
        }

        assertTrue(oriented, "Should have oriented colliders");

        // Check A -> B
        Link<Node> linkAB = probNet.getLink(nodeA, nodeB, true);
        assertNotNull(linkAB, "Should be A->B");
        assertTrue(linkAB.isDirected(), "A->B should be directed");

        // Check C -> B (Stored as Link C->B)
        Link<Node> linkCB = probNet.getLink(nodeC, nodeB, true);
        assertNotNull(linkCB, "Should be C->B");
        assertTrue(linkCB.isDirected(), "C->B should be directed");
    }
}
