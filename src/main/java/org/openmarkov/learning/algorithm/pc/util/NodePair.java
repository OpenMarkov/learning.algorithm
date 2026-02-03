package org.openmarkov.learning.algorithm.pc.util;

import org.openmarkov.core.model.network.Node;

/**
 * Immutable representation of an unordered pair of nodes.
 * <p>
 * Ensures that (X,Y) and (Y,X) are treated as equal by enforcing
 * a canonical order based on node names for deterministic behavior.
 */
public record NodePair(Node first, Node second) {

    /**
     * Compact canonical constructor that validates and enforces canonical ordering.
     * The nodes are ordered by name to ensure (A,B) and (B,A) are equivalent.
     */
    public NodePair {
        if (first == null || second == null) {
            throw new IllegalArgumentException("Nodes cannot be null.");
        }

        // Enforce canonical ordering: swap if needed
        int comparison = first.getName().compareTo(second.getName());
        if (comparison > 0) {
            // Swap to maintain canonical order
            Node temp = first;
            first = second;
            second = temp;
        }
    }

    /**
     * Factory method for creating a NodePair with automatic ordering.
     * This is the preferred way to create NodePair instances.
     * 
     * @param nodeA first node
     * @param nodeB second node
     * @return a NodePair with nodes in canonical order
     */
    public static NodePair of(Node nodeA, Node nodeB) {
        if (nodeA == null || nodeB == null) {
            throw new IllegalArgumentException("Nodes cannot be null.");
        }

        // Order by name for deterministic behavior
        int comparison = nodeA.getName().compareTo(nodeB.getName());
        if (comparison <= 0) {
            return new NodePair(nodeA, nodeB);
        } else {
            return new NodePair(nodeB, nodeA);
        }
    }

    @Override
    public String toString() {
        return "(" + first.getName() + ", " + second.getName() + ")";
    }
}
