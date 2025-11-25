package org.openmarkov.learning.algorithm.pc.util;

import org.openmarkov.core.model.network.Node;

/**
 * ExteriorImmutable representation of an unordered pair of nodes.
 * <p>
 * Ensures that (X,Y) and (Y,X) are treated as equal by enforcing
 * a canonical order based on {@link System#identityHashCode(Object)}.
 */
public record NodePair(Node first, Node second) {

    public NodePair {
        if (first == null || second == null) {
            throw new IllegalArgumentException("Nodes cannot be null.");
        }
        int hashA = System.identityHashCode(first);
        int hashB = System.identityHashCode(second);

        if (hashA > hashB || (hashA == hashB && first.hashCode() > second.hashCode())) {
            Node tmp = first;
            first = second;
            second = tmp;
        }
    }

    @Override
    public String toString() {
        return "(" + first.getName() + ", " + second.getName() + ")";
    }
}
