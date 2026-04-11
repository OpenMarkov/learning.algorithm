/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.pc;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.RemoveLinkEdit;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.learning.algorithm.pc.util.NodePair;
import org.openmarkov.learning.core.util.LearningEditProposal;

import java.util.*;

/**
 * Implements the skeleton discovery phase (INITIAL_PHASE) of the PC algorithm.
 * <p>
 * Uses the PC-Stable variant: adjacency snapshots are frozen at the start of each
 * conditioning-set depth level so that all pairs at depth d are tested with the same
 * candidates, regardless of which edges have been removed at that depth.
 * This makes the skeleton order-independent.
 *
 * @author Manuel Arias
 */
class SkeletonDiscovery {

    static final int ALREADY_DONE = -1;

    private final PCAlgorithm pc;

    /**
     * PC-Stable: current conditioning-set depth being explored.
     * Persists across getBestEdit calls so all removals at depth d are found before
     * advancing to depth d+1.
     */
    private int stableDepth = 0;

    /**
     * PC-Stable: snapshot of each node's neighbors at the start of {@link #stableDepth}.
     * Independence tests at depth d always use this frozen snapshot as the candidate
     * conditioning set, not the (potentially modified) live adjacency.
     * {@code null} means the snapshot has not been taken yet for the current depth.
     */
    private Map<Node, List<Node>> stableAdjSnapshot = null;

    private final Set<PNEdit> lastRemovedEdits = new HashSet<>();

    SkeletonDiscovery(PCAlgorithm pc) {
        this.pc = pc;
    }

    /**
     * Finds the best edit for the skeleton discovery phase.
     * <p>
     * Iterates over increasing conditioning-set depths using a frozen adjacency snapshot
     * per depth, so that removals within a depth do not affect the conditioning sets used
     * for other pairs at the same depth.
     *
     * @return the best removal proposal, or {@code null} when no more removals are possible
     */
    LearningEditProposal findEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        if (stableAdjSnapshot == null) {
            takeAdjacencySnapshot();
        }

        while (hasAnyPairAtDepth(stableDepth)) {
            separationSetsLogic(stableDepth);
            LearningEditProposal proposal = getOptimalEditFromCache(onlyAllowedEdits, onlyPositiveEdits);
            if (proposal != null) {
                return proposal;
            }
            stableDepth++;
            takeAdjacencySnapshot();
        }

        return null;
    }

    /**
     * Resets the PC-Stable skeleton state (depth counter and snapshot).
     * Must be called whenever the skeleton phase is restarted from scratch.
     */
    void resetState() {
        stableDepth = 0;
        stableAdjSnapshot = null;
    }

    void resetHistory() {
        lastRemovedEdits.clear();
    }

    /**
     * Called when a {@link RemoveLinkEdit} is executed: marks the pair as done in the cache,
     * invalidates neighbors whose conditioning set included the removed node, and resets
     * skeleton state so the next call restarts at depth 0 with a fresh snapshot.
     */
    void onEditExecuted(RemoveLinkEdit edit) {
        Node nodeX = pc.net().getNode(edit.getVariableFrom());
        Node nodeY = pc.net().getNode(edit.getVariableTo());

        PCEditMotivation cachedScore = pc.cache.get(new NodePair(nodeX, nodeY));
        List<Node> separationSet = cachedScore != null ? cachedScore.getSeparationSet() : new ArrayList<>();
        pc.cache.put(new NodePair(nodeX, nodeY), new PCEditMotivation(ALREADY_DONE, separationSet));

        // Invalidate cached tests whose separation set contained X or Y,
        // since those conditioning sets may no longer be subsets of the new adjacency.
        invalidateNeighborCache(nodeX, nodeY);
        invalidateNeighborCache(nodeY, nodeX);

        resetState();
    }

    /**
     * Called when a {@link RemoveLinkEdit} is undone: re-tests the pair and updates the cache.
     */
    void onEditUndone(RemoveLinkEdit edit) {
        Node nodeX = pc.net().getNode(edit.getVariableFrom());
        Node nodeY = pc.net().getNode(edit.getVariableTo());
        List<Node> separationSet = pc.cache.get(new NodePair(nodeX, nodeY)).getSeparationSet();
        double linkScore = pc.independenceTester.test(pc.database(), nodeX, nodeY, separationSet);
        pc.cache.put(new NodePair(nodeX, nodeY), new PCEditMotivation(linkScore, separationSet));
        resetState();
    }

    // ---- internal ----

    private void invalidateNeighborCache(Node node, Node removedNeighbor) {
        for (Node neighborNode : PCAlgorithm.sorted(node.getNeighbors())) {
            NodePair pair = new NodePair(node, neighborNode);
            PCEditMotivation neighborScore = pc.cache.get(pair);
            if (neighborScore != null && neighborScore.getScore() != ALREADY_DONE
                    && neighborScore.getSeparationSet().contains(removedNeighbor)) {
                pc.cache.remove(pair);
            }
        }
    }

    private void takeAdjacencySnapshot() {
        stableAdjSnapshot = new HashMap<>();
        for (Node node : pc.net().getNodes()) {
            stableAdjSnapshot.put(node, PCAlgorithm.sorted(node.getNeighbors()));
        }
    }

    /**
     * Returns true if there is at least one currently-present undirected edge (X-Y)
     * for which the snapshot conditioning set {@code snapshot[X] \ {Y}} has at least
     * {@code depth} elements and the edge has not yet been removed.
     */
    private boolean hasAnyPairAtDepth(int depth) {
        for (Node nodeX : pc.sortedNodes()) {
            List<Node> snapshotNeighbors = stableAdjSnapshot.getOrDefault(nodeX, Collections.emptyList());
            if (snapshotNeighbors.size() - 1 < depth) {
                continue;
            }
            for (Node nodeY : PCAlgorithm.sorted(nodeX.getSiblings())) {
                PCEditMotivation m = pc.cache.get(new NodePair(nodeX, nodeY));
                if (m == null || m.getScore() != ALREADY_DONE) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Iterates through all currently-present undirected edges (X-Y) and tests
     * independence at the given depth using the PC-Stable frozen snapshot.
     */
    private void separationSetsLogic(int adjacencySize) {
        for (Node nodeX : pc.sortedNodes()) {
            for (Node nodeY : PCAlgorithm.sorted(nodeX.getSiblings())) {
                List<Node> snapshotNeighbors = stableAdjSnapshot.getOrDefault(nodeX, Collections.emptyList());
                List<Node> adjacencySubset = new ArrayList<>(snapshotNeighbors);
                adjacencySubset.remove(nodeY);

                RemoveLinkEdit removeLinkEdit = new RemoveLinkEdit(
                        pc.net(), nodeX.getVariable(), nodeY.getVariable(), false);

                if (!pc.alreadyConsidered(removeLinkEdit, lastRemovedEdits)) {
                    PCEditMotivation motivation = pc.cache.get(new NodePair(nodeX, nodeY));
                    if (motivation == null || (motivation.getScore() != ALREADY_DONE
                            && motivation.getSeparationSet().size() < adjacencySize)) {
                        evaluateSeparationSets(nodeX, nodeY, adjacencySubset, adjacencySize);
                    }
                }
            }
        }
    }

    /**
     * Evaluates separation sets for a given pair of nodes and updates the cache.
     * Always caches the best separation set found regardless of the significance level;
     * the onlyPositiveEdits filtering is applied later when displaying the table.
     */
    private void evaluateSeparationSets(Node nodeX, Node nodeY,
                                        List<Node> adjacencySubset, int adjacencySize) {
        double bestScore = 0.0;
        List<Node> bestScoreSeparationSet = null;

        for (List<Node> separationSet : PCAlgorithm.subSetsOfSize(adjacencySubset, adjacencySize)) {
            double linkScore = pc.independenceTester.test(pc.database(), nodeX, nodeY, separationSet);
            if (linkScore > bestScore) {
                bestScore = linkScore;
                bestScoreSeparationSet = separationSet;
            }
        }

        if (bestScoreSeparationSet != null) {
            pc.cache.put(new NodePair(nodeX, nodeY),
                    new PCEditMotivation(bestScore, bestScoreSeparationSet));
        }
    }

    /**
     * Returns the optimal edit from the cache, according to the PC algorithm.
     */
    LearningEditProposal getOptimalEditFromCache(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        PCEditMotivation bestMotivation = null;
        LearningEditProposal bestEditProposal = null;

        for (Node nodeX : pc.sortedNodes()) {
            for (Node nodeY : PCAlgorithm.sorted(nodeX.getSiblings())) {
                PCEditMotivation motivation = pc.cache.get(new NodePair(nodeX, nodeY));
                if (!isCandidateMotivation(motivation, bestMotivation, onlyPositiveEdits)) {
                    continue;
                }

                RemoveLinkEdit removeLinkEdit =
                        new RemoveLinkEdit(pc.net(), nodeX.getVariable(), nodeY.getVariable(), false);

                if (isValidEdit(removeLinkEdit, bestMotivation, onlyAllowedEdits)) {
                    bestMotivation = motivation;
                    bestEditProposal = new LearningEditProposal(removeLinkEdit, motivation);
                }
            }
        }

        if (bestEditProposal != null) {
            lastRemovedEdits.add(bestEditProposal.getEdit());
        }

        return bestEditProposal;
    }

    private boolean isCandidateMotivation(PCEditMotivation motivation,
                                          PCEditMotivation bestMotivation,
                                          boolean onlyPositiveEdits) {
        if (motivation == null) {
            return false;
        }
        if (motivation.getScore() == ALREADY_DONE) {
            return false;
        }
        if (onlyPositiveEdits && motivation.getScore() <= pc.significanceLevel) {
            return false;
        }
        return bestMotivation == null || motivation.compareTo(bestMotivation) > 0;
    }

    private boolean isValidEdit(RemoveLinkEdit removeLinkEdit, PCEditMotivation bestMotivation,
                                boolean onlyAllowedEdits) {
        return !pc.checkBlocked(new LearningEditProposal(removeLinkEdit, bestMotivation))
                && !pc.alreadyConsidered(removeLinkEdit, lastRemovedEdits)
                && (!onlyAllowedEdits || pc.checkAllowed(removeLinkEdit));
    }
}
