/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.pc;

import org.openmarkov.core.action.core.COrientLinksEdit;
import org.openmarkov.core.action.base.linkEdits.OrientLinkEdit;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.learning.algorithm.pc.util.NodePair;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.StringEditMotivation;

import java.util.*;

/**
 * Implements the head-to-head orientation phase (collider detection) of the PC algorithm.
 * <p>
 * For every unconnected pair (X, Z) that share a common neighbor Y, if Y is not in the
 * separation set S(X, Z), then orient X->Y<-Z (a v-structure / collider).
 * <p>
 * This implementation correctly handles partially oriented graphs (e.g., A->B, B->E, C--E)
 * by orienting only the remaining undirected edges and avoiding redundant re-orientations.
 *
 * @author Manuel Arias
 */
class ColliderOrientation {

    private final PCAlgorithm pc;
    private final List<COrientLinksEdit> lastCompoundOrientationEdits = new ArrayList<>();

    ColliderOrientation(PCAlgorithm pc) {
        this.pc = pc;
    }

    /**
     * Detects and orients head-to-head (v-structure) patterns X->Y<-Z.
     * <p>
     * Iterates over every possible middle node Y and each unordered pair (X, Z) of Y's
     * neighbors. For each unshielded triple (X-Y-Z where X and Z are not adjacent),
     * checks if Y is absent from the separation set of X and Z. If so, orients toward Y.
     *
     * @param onlyAllowedEdits if true, only orientations allowed by structural constraints are considered
     * @return a {@link LearningEditProposal} with the orientation(s) to apply, or {@code null} if none found
     */
    LearningEditProposal findEdit(boolean onlyAllowedEdits) {
        for (Node nodeY : pc.sortedNodes()) {
            List<Node> neighborsY = PCAlgorithm.sorted(nodeY.getNeighbors());
            int n = neighborsY.size();

            for (int i = 0; i < n; i++) {
                Node nodeX = neighborsY.get(i);
                for (int j = i + 1; j < n; j++) {
                    Node nodeZ = neighborsY.get(j);

                    if (nodeX == nodeZ) {
                        continue;
                    }

                    // Unshielded triple condition: X and Z must NOT be adjacent
                    if (nodeX.getNeighbors().contains(nodeZ)) {
                        continue;
                    }

                    // Retrieve S(X, Z) from the cache (empty if not found)
                    List<Node> separationXZ = Optional.ofNullable(pc.cache.get(new NodePair(nodeX, nodeZ)))
                            .map(PCEditMotivation::getSeparationSet)
                            .orElse(Collections.emptyList());

                    // If Y not in S(X, Z), orient edges towards Y
                    if (!separationXZ.contains(nodeY)) {
                        LearningEditProposal proposal = tryOrientCollider(
                                nodeX, nodeY, nodeZ, onlyAllowedEdits);
                        if (proposal != null) {
                            return proposal;
                        }
                    }
                }
            }
        }

        return null;
    }

    boolean hasProducedEdits() {
        return !lastCompoundOrientationEdits.isEmpty();
    }

    void resetHistory() {
        lastCompoundOrientationEdits.clear();
    }

    // ---- internal ----

    /**
     * Attempts to create and return a compound orientation edit for the collider X->Y<-Z.
     * Only undirected edges are oriented; already-directed edges are skipped.
     *
     * @return the proposal, or {@code null} if the orientation is blocked, duplicate, or empty
     */
    private LearningEditProposal tryOrientCollider(Node nodeX, Node nodeY, Node nodeZ,
                                                   boolean onlyAllowedEdits) {
        OrientLinkEdit orientXY = new OrientLinkEdit(pc.net(),
                nodeX.getVariable(), nodeY.getVariable(), true);
        OrientLinkEdit orientZY = new OrientLinkEdit(pc.net(),
                nodeZ.getVariable(), nodeY.getVariable(), true);

        boolean allowedXY = pc.isOrientationAllowed(orientXY);
        boolean allowedZY = pc.isOrientationAllowed(orientZY);

        if (onlyAllowedEdits && !(allowedXY || allowedZY)) {
            return null;
        }

        // Collect only orientations that are still undirected (siblings)
        ArrayList<OrientLinkEdit> edits = new ArrayList<>();

        if (allowedXY && nodeX.isSibling(nodeY) && !createsContradictoryCollider(nodeX, nodeY)) {
            edits.add(orientXY);
        }
        if (allowedZY && nodeZ.isSibling(nodeY) && !createsContradictoryCollider(nodeZ, nodeY)) {
            edits.add(orientZY);
        }

        if (edits.isEmpty()) {
            return null;
        }

        COrientLinksEdit compoundEdit = new COrientLinksEdit(pc.net(), edits);
        StringEditMotivation motivation = new StringEditMotivation(
                "Sep. set (" + nodeX.getName() + ", " + nodeZ.getName() +
                        ") does not contain variable: " + nodeY.getName());

        LearningEditProposal proposal = new LearningEditProposal(compoundEdit, motivation);

        boolean duplicate = (edits.size() == 2) && alreadyConsidered(edits.get(0), edits.get(1));

        if (!duplicate && !pc.checkBlocked(proposal)) {
            lastCompoundOrientationEdits.add(compoundEdit);
            return proposal;
        }
        return null;
    }

    /**
     * Returns true if orienting {@code from -> to} would create an unshielded collider
     * {@code from -> to <- existingParent} that contradicts the separation set of
     * (from, existingParent).
     * <p>
     * A contradiction occurs when {@code to} belongs to {@code sep(from, existingParent)},
     * meaning the skeleton phase concluded that conditioning on {@code to} makes
     * {@code from} and {@code existingParent} independent (non-collider path).
     */
    private boolean createsContradictoryCollider(Node from, Node to) {
        for (Node existingParent : to.getParents()) {
            if (from.getNeighbors().contains(existingParent)) {
                continue;
            }
            PCEditMotivation sep = pc.cache.get(new NodePair(from, existingParent));
            if (sep != null && sep.getSeparationSet().contains(to)) {
                return true;
            }
            sep = pc.cache.get(new NodePair(existingParent, from));
            if (sep != null && sep.getSeparationSet().contains(to)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Checks whether a pair of OrientLinkEdits has already been considered as a compound edit.
     */
    private boolean alreadyConsidered(OrientLinkEdit edit1, OrientLinkEdit edit2) {
        boolean result = false;
        for (COrientLinksEdit compoundDirectLinkEdit : lastCompoundOrientationEdits) {
            result = result || (
                    (edit1.compareTo((OrientLinkEdit) compoundDirectLinkEdit.getEdits().findFirst().get()) == 0) && (
                            edit2.compareTo((OrientLinkEdit) compoundDirectLinkEdit.getEdits()
                                                                                   .skip(1)
                                                                                   .findFirst()
                                                                                   .get()) == 0
                    )
            );
            result = result || (
                    (edit1.compareTo((OrientLinkEdit) compoundDirectLinkEdit.getEdits()
                                                                            .skip(1)
                                                                            .findFirst()
                                                                            .get()) == 0) && (
                            edit2.compareTo((OrientLinkEdit) compoundDirectLinkEdit.getEdits().findFirst().get()) == 0
                    )
            );
        }
        return result;
    }
}
