/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.pc;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.OrientLinkEdit;
import org.openmarkov.core.model.graph.Link;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.StringEditMotivation;

import java.util.*;

/**
 * Implements the remaining-links orientation phase of the PC algorithm using
 * Meek's orientation rules (R1, R2, R3) followed by a fallback for truly
 * undetermined edges.
 * <p>
 * Meek (1995) proves that these rules, applied to closure, produce the unique CPDAG
 * (completed partially directed acyclic graph) representing the Markov equivalence
 * class of the true DAG.
 *
 * @author Manuel Arias
 */
class MeekOrientation {

    private final PCAlgorithm pc;
    private final Set<PNEdit> lastOrientationEdits = new HashSet<>();

    MeekOrientation(PCAlgorithm pc) {
        this.pc = pc;
    }

    /**
     * Orients remaining undirected links using Meek's rules, with fallback.
     *
     * @param onlyAllowedEdits whether to restrict to allowed orientations
     * @return a {@link LearningEditProposal} if an orientation is found; {@code null} otherwise
     */
    LearningEditProposal findEdit(boolean onlyAllowedEdits) {
        // R1: A->B-C, A not adjacent to C  =>  B->C
        LearningEditProposal p = meekR1(onlyAllowedEdits);
        if (p != null) return p;

        // R2: A->B->C, A-C  =>  A->C
        p = meekR2(onlyAllowedEdits);
        if (p != null) return p;

        // R3: D-A, D-B, D-C, B->A, C->A, B not adjacent to C  =>  D->A
        p = meekR3(onlyAllowedEdits);
        if (p != null) return p;

        // Fallback: orient using ANM if available, else arbitrary (preserves acyclicity)
        return tryOrientUnorientedWithoutPath(onlyAllowedEdits);
    }

    boolean hasProducedEdits() {
        return !lastOrientationEdits.isEmpty();
    }

    void resetHistory() {
        lastOrientationEdits.clear();
    }

    // ---- Meek R1 ----

    /**
     * Meek Rule R1: for each undirected edge B-C, if there exists A->B where A and C
     * are not adjacent, orient B->C.
     * <p>
     * Rationale: orienting C->B instead would create a new unshielded collider A->B<-C
     * (unshielded because A and C are not adjacent), contradicting the collider set.
     */
    private LearningEditProposal meekR1(boolean onlyAllowedEdits) {
        for (Link<Node> link : pc.sortedLinks()) {
            if (!link.isDirected()) {
                LearningEditProposal p = meekR1Orient(link.getFrom(), link.getTo(), onlyAllowedEdits);
                if (p != null) return p;
                p = meekR1Orient(link.getTo(), link.getFrom(), onlyAllowedEdits);
                if (p != null) return p;
            }
        }
        return null;
    }

    private LearningEditProposal meekR1Orient(Node nodeB, Node nodeC, boolean onlyAllowedEdits) {
        for (Node nodeA : PCAlgorithm.sorted(nodeB.getParents())) {
            if (!nodeA.getNeighbors().contains(nodeC)) {
                LearningEditProposal proposal = buildOrientProposal(nodeB, nodeC, "Meek R1", onlyAllowedEdits);
                if (proposal != null) return proposal;
            }
        }
        return null;
    }

    // ---- Meek R2 ----

    /**
     * Meek Rule R2: for each undirected edge A-C, if there exists a directed path A=>C,
     * orient A->C.
     * <p>
     * Rationale: orienting C->A instead would create a directed cycle.
     */
    private LearningEditProposal meekR2(boolean onlyAllowedEdits) {
        for (Link<Node> link : pc.sortedLinks()) {
            if (!link.isDirected()) {
                LearningEditProposal p = meekR2Orient(link.getFrom(), link.getTo(), onlyAllowedEdits);
                if (p != null) return p;
                p = meekR2Orient(link.getTo(), link.getFrom(), onlyAllowedEdits);
                if (p != null) return p;
            }
        }
        return null;
    }

    private LearningEditProposal meekR2Orient(Node nodeA, Node nodeC, boolean onlyAllowedEdits) {
        if (pc.net().existsPath(nodeA, nodeC, true, Collections.emptyList())) {
            return buildOrientProposal(nodeA, nodeC, "Meek R2", onlyAllowedEdits);
        }
        return null;
    }

    // ---- Meek R3 ----

    /**
     * Meek Rule R3: for each undirected edge D-A, if there exist two distinct nodes B and C
     * such that D-B (undirected), D-C (undirected), B->A, C->A, and B not adjacent to C,
     * orient D->A.
     * <p>
     * Rationale: orienting A->D instead would create a new unshielded collider at D
     * in the path B->A->D<-C, since B and C are not adjacent.
     */
    private LearningEditProposal meekR3(boolean onlyAllowedEdits) {
        for (Link<Node> link : pc.sortedLinks()) {
            if (!link.isDirected()) {
                LearningEditProposal p = meekR3Orient(link.getFrom(), link.getTo(), onlyAllowedEdits);
                if (p != null) return p;
                p = meekR3Orient(link.getTo(), link.getFrom(), onlyAllowedEdits);
                if (p != null) return p;
            }
        }
        return null;
    }

    private LearningEditProposal meekR3Orient(Node nodeD, Node nodeA, boolean onlyAllowedEdits) {
        // Candidates: parents of A that are also undirected siblings of D
        List<Node> candidates = new ArrayList<>(nodeA.getParents());
        candidates.retainAll(nodeD.getSiblings());
        candidates.sort(Comparator.comparing(Node::getName));

        for (int i = 0; i < candidates.size(); i++) {
            Node nodeB = candidates.get(i);
            for (int j = i + 1; j < candidates.size(); j++) {
                Node nodeC = candidates.get(j);
                if (!nodeB.getNeighbors().contains(nodeC)) {
                    LearningEditProposal proposal = buildOrientProposal(
                            nodeD, nodeA, "Meek R3", onlyAllowedEdits);
                    if (proposal != null) return proposal;
                }
            }
        }
        return null;
    }

    // ---- Fallback ----

    /**
     * Attempts to orient undirected links (X-Z) when no Meek rule applies,
     * as a last resort. Uses the ANM causal direction tester if available,
     * otherwise orients arbitrarily while preserving acyclicity.
     */
    private LearningEditProposal tryOrientUnorientedWithoutPath(boolean onlyAllowedEdits) {
        for (Link<Node> link : pc.sortedLinks()) {
            if (!link.isDirected()) {
                Node nodeX = link.getFrom();
                Node nodeZ = link.getTo();

                // Determine preferred direction via ANM tester if available
                Node preferred = nodeX;
                Node other     = nodeZ;
                String motivation = "Do not create cycles";
                if (pc.causalDirectionTester != null) {
                    double scoreXZ = pc.causalDirectionTester.testDirection(pc.database(), nodeX, nodeZ);
                    double scoreZX = pc.causalDirectionTester.testDirection(pc.database(), nodeZ, nodeX);
                    if (scoreZX > scoreXZ) {
                        preferred = nodeZ;
                        other     = nodeX;
                    }
                    motivation = "Causal direction test (ANM)";
                }

                // Try preferred direction first
                LearningEditProposal proposal = tryOrientDirection(
                        preferred, other, motivation, onlyAllowedEdits);
                if (proposal != null) return proposal;

                // Fall back to opposite direction
                proposal = tryOrientDirection(
                        other, preferred, "Do not create cycles", onlyAllowedEdits);
                if (proposal != null) return proposal;
            }
        }
        return null;
    }

    /**
     * Tries to orient {@code from -> to}, checking for cycles, duplicates, and constraints.
     */
    private LearningEditProposal tryOrientDirection(Node from, Node to,
                                                    String motivation, boolean onlyAllowedEdits) {
        OrientLinkEdit edit = new OrientLinkEdit(pc.net(), from.getVariable(), to.getVariable(), true);
        LearningEditProposal proposal = new LearningEditProposal(edit, new StringEditMotivation(motivation));
        if (!pc.net().existsPath(to, from, true, Collections.emptyList())
                && !pc.alreadyConsidered(edit, lastOrientationEdits)
                && !pc.checkBlocked(proposal)
                && (!onlyAllowedEdits || pc.isOrientationAllowed(edit))) {
            lastOrientationEdits.add(edit);
            return proposal;
        }
        return null;
    }

    // ---- Common ----

    /**
     * Builds and validates an {@link OrientLinkEdit} proposal for orienting {@code from->to}.
     * Returns {@code null} if the edit was already considered, is blocked, or is not allowed.
     */
    private LearningEditProposal buildOrientProposal(Node from, Node to,
                                                     String motivation, boolean onlyAllowedEdits) {
        OrientLinkEdit edit = new OrientLinkEdit(pc.net(), from.getVariable(), to.getVariable(), true);
        LearningEditProposal proposal = new LearningEditProposal(edit, new StringEditMotivation(motivation));
        if (!pc.alreadyConsidered(edit, lastOrientationEdits)
                && !pc.checkBlocked(proposal)
                && (!onlyAllowedEdits || pc.isOrientationAllowed(edit))) {
            lastOrientationEdits.add(edit);
            return proposal;
        }
        return null;
    }
}
