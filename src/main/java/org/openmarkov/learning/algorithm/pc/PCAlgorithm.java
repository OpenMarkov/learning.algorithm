/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.pc;

import org.openmarkov.core.action.*;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.graph.Link;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.learning.algorithm.pc.independencetester.IndependenceTester;
import org.openmarkov.learning.algorithm.pc.util.PCEditMotivation;
import org.openmarkov.learning.core.algorithm.LearningAlgorithmType;
import org.openmarkov.learning.core.util.LearningEditMotivation;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.StringEditMotivation;
import org.openmarkov.learning.algorithm.pc.util.NodePair;

import javax.swing.event.UndoableEditEvent;
import javax.swing.undo.UndoableEdit;
import java.util.*;

/**
 * PC (Peter-Clark) Algorithm for learning Bayesian Network structure.
 * This algorithm uses conditional independence tests to discover
 * the causal structure of a Bayesian Network.<p>
 * The algorithm works in three main phases:
 * <ol>
 * <li>INITIAL_PHASE: Discovering independence relations and removing links</li>
 * <li>HEAD_TO_HEAD_ORIENTATION: Orienting colliders (head-to-head links)</li>
 * <li>REMAINING_LINKS_ORIENTATION: Orienting remaining links to maintain DAG structure</li>
 * </ol>
 */
@LearningAlgorithmType(name = "PC", discriminative = false, supportsUnobservedVariables = false)
public class PCAlgorithm extends IndependenceRelationsAlgorithm
        implements PNUndoableEditListener {
    
    // Constants
    private static final int ALREADY_DONE = -1;
    
    // Algorithm phases
    private enum Phase {
        INITIAL_PHASE,
        HEAD_TO_HEAD_ORIENTATION,
        REMAINING_LINKS_ORIENTATION,
        ORIENTATION_FINISHED
    }
    
    // Core algorithm components
    /**
     * Cache for storing independence test results between nodes
     */
    protected final Map<NodePair, PCEditMotivation> cache;
    
    /**
     * History of last best edits returned.
     */
    protected final Set<PNEdit> lastRemovedEdits = new HashSet<>();
    
    /**
     * History of last best edits returned.
     */
    protected final Set<PNEdit> lastOrientationEdits = new HashSet<>();
    
    /**
     * History of last best edits returned.
     */
    protected final List<COrientLinksEdit> lastCompoundOrientationEdits = new ArrayList<>();
    
    protected IndependenceTester independenceTester;
    
    /**
     * Degree of accuracy of the independence test.
     */
    protected double significanceLevel;
    
    /**
     * Current algorithm phase
     */
    private Phase phase;
    
    /**
     * Constructor for the PC Algorithm
     *
     * @param probNet            Probabilistic Network to learn, initially it contains only the nodes.
     * @param caseDatabase       Database of cases
     * @param alpha              Learning rate
     * @param independenceTester Independence test method
     * @param significanceLevel  Statistical significance level
     */
    public PCAlgorithm(
    		ProbNet probNet, 
    		CaseDatabase caseDatabase, 
    		Double alpha, 
    		IndependenceTester independenceTester,
            Double significanceLevel) {
    	
        super(probNet, caseDatabase, alpha);
        this.independenceTester = independenceTester;
        this.significanceLevel = significanceLevel;
        this.probNet.getPNESupport().addUndoableEditListener(this);
        
        cache = new HashMap<>();
        
        this.phase = Phase.INITIAL_PHASE;
    }
    
    /**
     * Method that returns the best edit in each step of the algorithm or null
     * if there are no more edits to consider.
     *
     * @param onlyAllowedEdits
     * @param onlyPositiveEdits
     * @return LearningEditProposal, or null if no edits are available.
     */
    @Override
    public LearningEditProposal getBestEdit(
    		boolean onlyAllowedEdits, 
    		boolean onlyPositiveEdits) {
        
    	resetHistory();
        
        return getNextEdit(onlyAllowedEdits, onlyPositiveEdits);
    }
    
    /**
     * Method that returns the next best edit in each step of the algorithm
     * or null if there are no more edits to consider (depending on the
     * arguments it receives).
     *
     * @param onlyAllowedEdits  if true, only allowed edits are considered
     * @param onlyPositiveEdits if true, only positive edits are considered
     * @return LearningEditProposal
     */
    @Override
    public LearningEditProposal getNextEdit(
    		boolean onlyAllowedEdits, 
    		boolean onlyPositiveEdits) {
        
        LearningEditProposal bestEditProposal;
        do {
            bestEditProposal = getOptimalEdit(onlyAllowedEdits, onlyPositiveEdits);
        } while (bestEditProposal != null && isBlocked(bestEditProposal)); // Skip blocked edits
        return bestEditProposal;
    }
    
    /**
     * Finds the optimal edit based on current algorithm phase and adjacency size.
     *
     * @param onlyAllowedEdits  if true, only allowed edits are considered
     * @param onlyPositiveEdits if true, only positive edits are considered
     * @return LearningEditProposal
     */
    public LearningEditProposal getOptimalEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        int adjacencySize = 0;
        LearningEditProposal bestEditProposal;
        
        while (maxOfAdjacencies() > adjacencySize) {
            bestEditProposal = findBestEditInCurrentPhase(adjacencySize, onlyAllowedEdits, onlyPositiveEdits);
            if (bestEditProposal != null) {
                return bestEditProposal;
            }
            adjacencySize++;
        }
        
        // bestEditProposal == null. Transition between phases
        return transitionToNextPhase(onlyAllowedEdits);
    }
    
    /**
     * Find the best edit by evaluating separation sets for node pairs.
     * This method iterates through all nodes and their neighbors,
     * calculating the best separation set for each pair.
     *
     * @param adjacencySize
     * @param onlyAllowedEdits
     * @param onlyPositiveEdits
     * @return
     */
    private LearningEditProposal findBestEditInCurrentPhase(int adjacencySize,
                                                            boolean onlyAllowedEdits,
                                                            boolean onlyPositiveEdits) {
        
        return switch (phase) {
            case INITIAL_PHASE -> {
                separationSetsLogic(adjacencySize, onlyPositiveEdits);
                yield getOptimalEditFromCache(onlyAllowedEdits, onlyPositiveEdits);
            }
            case HEAD_TO_HEAD_ORIENTATION -> getOrientationEdit(onlyAllowedEdits);
            case REMAINING_LINKS_ORIENTATION -> orientRemainingLinks(onlyAllowedEdits);
            default -> null;
        };
    }
    
    /**
     * Logic for evaluating separation sets in the INITIAL_PHASE.
     * This method iterates through all nodes and their neighbors,
     * calculating the best separation set for each pair of nodes.
     *
     * @param adjacencySize     Size of the adjacency set to consider
     * @param onlyPositiveEdits If true, only positive edits are considered
     */
    private void separationSetsLogic(int adjacencySize, boolean onlyPositiveEdits) {
        // Existing separation-set evaluation logic
        for (Node nodeX : probNet.getNodes()) {
            for (Node nodeY : nodeX.getSiblings()) {
                List<Node> adjacencySubset = new ArrayList<>(nodeX.getNeighbors());
                adjacencySubset.remove(nodeY);
                
                RemoveLinkEdit removeLinkEdit = new RemoveLinkEdit(
                        probNet, nodeX.getVariable(), nodeY.getVariable(), false);
                
                if (!alreadyConsidered(removeLinkEdit, lastRemovedEdits)) {
                	PCEditMotivation motivation = cache.get(new NodePair(nodeX, nodeY));
                    
                    // Evaluate separation sets if not already cached or needs recalculation
                    if (motivation == null || (motivation.getScore() != ALREADY_DONE
                            && motivation.getSeparationSet().size() > adjacencySize)) {
                        evaluateSeparationSets(nodeX, nodeY, adjacencySubset, adjacencySize, onlyPositiveEdits);
                    }
                }
            }
        }
        
    }
    
    /**
     * Transition to next phase when no more edits are possible
     *
     * @param onlyAllowedEdits
     * @return
     */
    private LearningEditProposal transitionToNextPhase(boolean onlyAllowedEdits) {
        if (lastRemovedEdits.isEmpty()) {
            phase = Phase.HEAD_TO_HEAD_ORIENTATION;
            return getOrientationEdit(onlyAllowedEdits);
        }
        phase = Phase.INITIAL_PHASE;
        return null;
    }
    
    /**
     * Evaluates separation sets for a given pair of nodes and updates the cache.
     *
     * @param nodeX
     * @param nodeY
     * @param adjacencySubset
     * @param adjacencySize
     * @param onlyPositiveEdits
     */
    private void evaluateSeparationSets(Node nodeX, Node nodeY, List<Node> adjacencySubset, int adjacencySize, boolean onlyPositiveEdits) {
        double bestScore = 0.0;
        List<Node> bestScoreSeparationSet = null;
        
        for (List<Node> separationSet : subSetsOfSize(adjacencySubset, adjacencySize)) {
            double linkScore = independenceTester.test(caseDatabase, nodeX, nodeY, separationSet);
            if (linkScore > bestScore && (!onlyPositiveEdits || linkScore > significanceLevel)) {
                bestScore = linkScore;
                bestScoreSeparationSet = separationSet;
            }
        }
        
        if (bestScoreSeparationSet != null) {
            cache.put(new NodePair(nodeX, nodeY), new PCEditMotivation(bestScore, bestScoreSeparationSet));
        }
    }
    
    /**
     * Returns the optimal edit from the cache, according to the PC algorithm.
     *
     * @param onlyAllowedEdits  If true, only edits allowed by current constraints are considered.
     * @param onlyPositiveEdits If true, only edits with score greater than the significance level are considered.
     * @return The best {@link LearningEditProposal}, or {@code null} if none is found.
     */
    public LearningEditProposal getOptimalEditFromCache(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        PCEditMotivation bestMotivation = null;
        LearningEditProposal bestEditProposal = null;

        for (Node nodeX : probNet.getNodes()) {

            for (Node nodeY : nodeX.getSiblings()) {
            	PCEditMotivation motivation = cache.get(new NodePair(nodeX, nodeY));
                if (!isCandidateMotivation(motivation, bestMotivation, onlyPositiveEdits)) {
                    continue;
                }

                RemoveLinkEdit removeLinkEdit =
                        new RemoveLinkEdit(probNet, nodeX.getVariable(), nodeY.getVariable(), false);

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

    /**
     * Checks whether a given motivation is a valid candidate to replace the current best.
     */
    private boolean isCandidateMotivation(PCEditMotivation motivation,
                                          PCEditMotivation bestMotivation,
                                          boolean onlyPositiveEdits) {
        if (motivation == null) {
            return false;
        }
        if (motivation.getScore() == ALREADY_DONE) {
            return false;
        }
        if (onlyPositiveEdits && motivation.getScore() <= significanceLevel) {
            return false;
        }
        return bestMotivation == null || motivation.compareTo(bestMotivation) > 0;
    }

    
    /**
     * @param removeLinkEdit
     * @param bestMotivation
     * @param onlyAllowedEdits
     * @return true if the edit is valid, false otherwise
     */
    private boolean isValidEdit(RemoveLinkEdit removeLinkEdit, PCEditMotivation bestMotivation,
                                boolean onlyAllowedEdits) {
        return !isBlocked(new LearningEditProposal(removeLinkEdit, bestMotivation)) &&
                !alreadyConsidered(removeLinkEdit, lastRemovedEdits) &&
                (!onlyAllowedEdits || isAllowed(removeLinkEdit));
    }
    
    /**
     * Returns the {@code PCEditProposal} with the
     * {@code DirectLinkEdit} depending on which stage is the algorithm.
     * If the "head to head" orientations have not been done, then, the
     * DirectLinkEdit contains these edits. Else, it contains the remaining
     * orientations.
     *
     * @param onlyAllowedEdits
     * @return LearningEditProposal the orientation edit
     */
    public LearningEditProposal getOrientationEdit(boolean onlyAllowedEdits) {
        LearningEditProposal bestEdit = orientHeadToHeadLinks(onlyAllowedEdits);
        if (bestEdit == null) {
            if (lastCompoundOrientationEdits.isEmpty()) {
                phase = Phase.REMAINING_LINKS_ORIENTATION;
                return orientRemainingLinks(onlyAllowedEdits);
            }
        }
        return bestEdit;
    }
    
    /**
     * @return int The number of neighbors of the node with the maximum
     */
    private int maxOfAdjacencies() {
        int max = 0;
        for (Node node : probNet.getNodes()) {
            int adjacents = node.getNumNeighbors();
            if (adjacents > max)
                max = adjacents;
        }
        return max;
    }
    
    /**
     * Returns a list of the subsets of size n of the given set
     *
     * @param set         {@code List} of
     *                    {@code Node} from which extract the subsets.
     * @param subSetsSize size of the subsets.
     * @return {@code List} of {@code List} of
     * {@code Node}. Each {@code List} of {@code Node}
     * is one of the subsets of size n.
     */
    public static List<List<Node>> subSetsOfSize(List<Node> set, int subSetsSize) {
        
        List<List<Node>> subSets = new ArrayList<>();
        List<Node> subSet = new ArrayList<>();
        boolean found = true;
        int[] indexSubSet = new int[subSetsSize];
        
        //Add the empty set
        if (subSetsSize == 0) {
            subSets.add(new ArrayList<>());
        }
        
        if ((subSetsSize > 0) && (subSetsSize <= set.size())) {
            for (int i = 0; i < subSetsSize; i++) {
                indexSubSet[i] = i;
                subSet.add(set.get(i));
            }
            subSets.add(subSet);
            
            if (subSetsSize < set.size()) {
                while (found) {
                    found = false;
                    
                    for (int i = subSetsSize - 1; i >= 0; i--) {
                        if (indexSubSet[i] < (set.size() + (i - subSetsSize))) {
                            indexSubSet[i] = indexSubSet[i] + 1;
                            
                            if (i < (subSetsSize - 1)) {
                                for (int j = i + 1; j < subSetsSize; j++) {
                                    indexSubSet[j] = indexSubSet[j - 1] + 1;
                                }
                            }
                            
                            found = true;
                            break;
                        }
                    }
                    
                    if (found) {
                        subSet = new ArrayList<>();
                        for (int k = 0; k < subSetsSize; k++) {
                            subSet.add(set.get(indexSubSet[k]));
                        }
                        
                        subSets.add(subSet);
                    }
                }
            }
        }
        
        return subSets;
    }
    
    /**
     * Given a RemoveLinkEdit, this method returns the same link with the inverse
     * direction. For example, if the parameter edit is a RemoveLinkEdit A-&gt;B,
     * it returns the RemoveLinkEdit B-&gt;A
     *
     * @param edit RemoveLinkEdit to be inverted
     * @return RemoveLinkEdit with the inverse direction
     */
    public RemoveLinkEdit inverseEdit(RemoveLinkEdit edit) {
        return new RemoveLinkEdit(probNet, edit.getVariable2(), edit.getVariable1(), false);
    }
    
    /**
     * @param edit
     * @param consideredEdits
     * @return true if the edit has already been considered, false otherwise
     */
    public boolean alreadyConsidered(BaseLinkEdit edit, Set<PNEdit> consideredEdits) {
        BaseLinkEdit inverseEdit = new RemoveLinkEdit(probNet, edit.getVariable2(), edit.getVariable1(),
                                                      edit.isDirected());
        return consideredEdits.contains(edit) || consideredEdits.contains(inverseEdit);
    }
    
    /**
     * @param edit1
     * @param edit2
     * @return true if the edits have already been considered, false otherwise
     */
    public boolean alreadyConsidered(OrientLinkEdit edit1, OrientLinkEdit edit2) {
        boolean result = false;
        
        for (COrientLinksEdit compoundDirectLinkEdit : lastCompoundOrientationEdits) {
            result |= (
                    (edit1.compareTo((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(0)) == 0) && (
                            edit2.compareTo((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(1)) == 0
                    )
            );
            result |= (
                    (edit1.compareTo((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(1)) == 0) && (
                            edit2.compareTo((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(0)) == 0
                    )
            );
        }
        return result;
    }
    
    /**
     * Detects and orients head-to-head (v-structure) patterns X->Y<-Z according to the PC algorithm rule:
     * For every unconnected pair (X, Z) that share a common neighbor Y, if Y ∉ S(X, Z)
     * (the separation set of X and Z), then orient X->Y<-Z.
     *
     * This implementation correctly handles partially oriented graphs (e.g., A->B, B->E, C--E)
     * by orienting only the remaining undirected edges and avoiding redundant re-orientations.
     *
     * Main design decisions:
     *  - Iterate over Y and all its general neighbors (directed or undirected) using getNeighbors().
     *  - Do NOT remove X from Y’s neighborhood list; this avoids missing valid triplets in mixed graphs.
     *  - Add to the compound edit only the orientations that are still undirected (isSibling()).
     *  - Skip triples where both candidate links are already directed.
     *
     * @param onlyAllowedEdits if true, only orientations allowed by structural constraints are considered
     * @return a LearningEditProposal with the orientation(s) to apply, or null if none found
     */
    private LearningEditProposal orientHeadToHeadLinks(boolean onlyAllowedEdits) {

        COrientLinksEdit compoundDirectLinkEdit;
        StringEditMotivation stringMotivation;

        // Iterate over every possible middle node Y in a potential X–Y–Z triple
        for (Node nodeY : probNet.getNodes()) {

            // Obtain all neighbors of Y (both directed and undirected)
            List<Node> neighborsY = new ArrayList<>(nodeY.getNeighbors());
            int n = neighborsY.size();

            // For each unordered pair (X, Z) of Y's neighbors
            for (int i = 0; i < n; i++) {
                Node nodeX = neighborsY.get(i);
                for (int j = i + 1; j < n; j++) {
                    Node nodeZ = neighborsY.get(j);

                    if (nodeX == nodeZ) {
                        continue; // safety check
                    }

                    // Ensure X and Z are NOT adjacent (unshielded triple condition)
                    if (nodeX.getNeighbors().contains(nodeZ)) {
                        continue;
                    }

                    // Retrieve the separation set S(X, Z) from the cache (empty if not found)
                    List<Node> separationXZ = Optional.ofNullable(cache.get(new NodePair(nodeX, nodeZ)))
                            .map(PCEditMotivation::getSeparationSet)
                            .orElse(Collections.emptyList());

                    // If Y ∉ S(X, Z), we must orient edges towards Y (X->Y<-Z)
                    if (!separationXZ.contains(nodeY)) {

                        // Prepare orientation edits towards Y
                        OrientLinkEdit orientXY = new OrientLinkEdit(probNet,
                                nodeX.getVariable(), nodeY.getVariable(), true);
                        OrientLinkEdit orientZY = new OrientLinkEdit(probNet,
                                nodeZ.getVariable(), nodeY.getVariable(), true);

                        // Check if each orientation is allowed (according to current constraints)
                        boolean allowedXY = isOrientationAllowed(orientXY);
                        boolean allowedZY = isOrientationAllowed(orientZY);

                        // If both are forbidden under the constraint mode, skip this triple
                        if (onlyAllowedEdits && !(allowedXY || allowedZY)) {
                            continue;
                        }

                        // Collect only orientations that are still undirected (siblings)
                        Vector<OrientLinkEdit> edits = new Vector<>();

                        if (allowedXY && nodeX.isSibling(nodeY)) {
                            // X–Y is undirected: orient X->Y
                            edits.add(orientXY);
                        }
                        if (allowedZY && nodeZ.isSibling(nodeY)) {
                            // Z–Y is undirected: orient Z->Y (critical in A->B, B->E, C--E)
                            edits.add(orientZY);
                        }

                        // If both edges are already directed, skip this case
                        if (edits.isEmpty()) {
                            continue;
                        }

                        // Create a compound edit for all required orientations
                        compoundDirectLinkEdit = new COrientLinksEdit(probNet, edits);

                        // Explanation text for logging and traceability
                        stringMotivation = new StringEditMotivation(
                                "Sep. set (" + nodeX.getName() + ", " + nodeZ.getName() +
                                        ") does not contain variable: " + nodeY.getName());

                        LearningEditProposal proposal =
                                new LearningEditProposal(compoundDirectLinkEdit, stringMotivation);

                        // Avoid duplicates or blocked proposals
                        boolean duplicate =
                                (edits.size() == 2) && alreadyConsidered((OrientLinkEdit)edits.get(0), (OrientLinkEdit)edits.get(1));

                        if (!duplicate && !isBlocked(proposal)) {
                            lastCompoundOrientationEdits.add(compoundDirectLinkEdit);
                            return proposal; // return the first applicable proposal
                        }
                    }
                }
            }
        }

        // No applicable orientation found
        return null;
    }   
    /**
     * Method to compute the final stage of the algorithm. The basic idea is
     * that no new head-to-head links are created and that the DAG condition is
     * preserved.
     *
     */
    private LearningEditProposal orientRemainingLinks(boolean onlyAllowedEdits) {
        
        // First pass: Try to orient links based on existing directed links (X → Z)
        LearningEditProposal editProposal = tryOrientFromDirectedLinks(onlyAllowedEdits);
        if (editProposal != null) 
        	return editProposal;
        
        // Second pass: Try to orient links based on existing paths (X—Z with path X→Z or Z→X)
        editProposal = tryOrientFromNonOrientedLinks(onlyAllowedEdits);
        if (editProposal != null) 
        	return editProposal;
        
        // Third pass: Try to orient non-directed links where no path exists in either direction
        editProposal = tryOrientUnorientedWithoutPath(onlyAllowedEdits);
        if (editProposal != null) 
        	return editProposal;
        
        // No valid orientation found; mark phase as finished if no edits were proposed
        if (lastOrientationEdits.isEmpty()) {
            phase = Phase.ORIENTATION_FINISHED;
        }
        return null;
    }
    
    /**
     * Attempts to orient links based on existing directed links (X → Z).
     * If a structure X → Z — Y is found and X is not adjacent to Y,
     * it proposes to orient Z — Y to avoid introducing cycles.
     *
     * @param onlyAllowedEdits whether to restrict to allowed orientations
     * @return a LearningEditProposal if an orientation is possible; null otherwise
     */
    private LearningEditProposal tryOrientFromDirectedLinks(boolean onlyAllowedEdits) {
        for (Link<Node> link : probNet.getLinks()) {
            if (link.isDirected()) {
                Node nodeX = link.getNode1();
                Node nodeZ = link.getNode2();
                for (Node nodeY : nodeZ.getSiblings()) {
                    OrientLinkEdit edit = new OrientLinkEdit(probNet, nodeZ.getVariable(), nodeY.getVariable(), true);
                    LearningEditProposal proposal = new LearningEditProposal(edit, new StringEditMotivation("Do not create cycles"));
                    
                    // Conditions:
                    // 1. X is not adjacent to Y
                    // 2. This orientation hasn't already been tried
                    // 3. The orientation is not blocked
                    // 4. If filtering is active, the orientation must be allowed
                    if (!nodeY.getNeighbors().contains(nodeX)
                            && !alreadyConsidered(edit, lastOrientationEdits)
                            && !isBlocked(proposal)
                            && (!onlyAllowedEdits || isOrientationAllowed(edit))) {
                        lastOrientationEdits.add(edit);
                        return proposal;
                    }
                }
            }
        }
        return null;
    }
    
    /**
     * Attempts to orient non-directed links (X—Z) based on the existence of directed paths
     * from X to Z or from Z to X, or based on collider patterns between neighbors of Z.
     *
     * @param onlyAllowedEdits whether to restrict to allowed orientations
     * @return a LearningEditProposal if an orientation is possible; null otherwise
     */
    private LearningEditProposal tryOrientFromNonOrientedLinks(boolean onlyAllowedEdits) {
        for (Link<Node> link : probNet.getLinks()) {
            if (!link.isDirected()) {
                Node nodeX = link.getNode1();
                Node nodeZ = link.getNode2();
                
                LearningEditProposal proposal = tryOrientIfPathExists(nodeX, nodeZ, onlyAllowedEdits);
                if (proposal != null) return proposal;
                
                proposal = tryOrientIfPathExists(nodeZ, nodeX, onlyAllowedEdits);
                if (proposal != null) return proposal;
                
                proposal = tryColliderPatterns(nodeX, nodeZ, onlyAllowedEdits);
                if (proposal != null) return proposal;
            }
        }
        return null;
    }
    
    /**
     * Tries to orient a non-directed link if a directed path exists between the nodes.
     *
     * @param from             starting node
     * @param to               target node
     * @param onlyAllowedEdits whether to restrict to allowed orientations
     * @return a LearningEditProposal if the path justifies the orientation; null otherwise
     */
    private LearningEditProposal tryOrientIfPathExists(Node from, Node to, boolean onlyAllowedEdits) {
        OrientLinkEdit edit = new OrientLinkEdit(probNet, from.getVariable(), to.getVariable(), true);
        LearningEditProposal proposal = new LearningEditProposal(edit, new StringEditMotivation("Do not create cycles"));
        
        // Conditions:
        // 1. There is a directed path from 'from' to 'to' in the DAG
        // 2. This orientation hasn't already been proposed
        // 3. It is not blocked by any constraint
        // 4. If restrictions apply, it must be allowed
        if (probNet.existsPath(from, to, true)
                && !alreadyConsidered(edit, lastOrientationEdits)
                && !isBlocked(proposal)
                && (!onlyAllowedEdits || isOrientationAllowed(edit))) {
            lastOrientationEdits.add(edit);
            return proposal;
        }
        return null;
    }
    
    /**
     * Attempts to orient links based on collider-like patterns (Y—Z—W),
     * looking for triplets that form converging arrows (e.g., Y → Z ← W)
     * to prevent unshielded colliders and cycles.
     *
     * @param nodeX            the node not adjacent to Y or W
     * @param nodeZ            the common neighbor
     * @param onlyAllowedEdits whether to restrict to allowed orientations
     * @return a LearningEditProposal if a collider orientation is proposed; null otherwise
     */
    private LearningEditProposal tryColliderPatterns(Node nodeX, Node nodeZ, boolean onlyAllowedEdits) {
        List<Node> siblingsZ = new ArrayList<>(nodeZ.getSiblings());
        siblingsZ.remove(nodeX);
        for (Node nodeY : siblingsZ) {
            if (!nodeY.getNeighbors().contains(nodeX)) {
                for (Node nodeW : siblingsZ) {
                    if (!nodeY.equals(nodeW)) {
                        // Skip this combination if:
                        // - X is not a parent of W
                        // - or Z already has an oriented link to Y
                        boolean skip = !nodeX.isParent(nodeW) || probNet.getLink(nodeZ, nodeY, true) != null;
                        
                        if (!skip && nodeY.getChildren().contains(nodeW)) {
                            OrientLinkEdit edit = new OrientLinkEdit(probNet, nodeZ.getVariable(), nodeW.getVariable(), true);
                            LearningEditProposal proposal = new LearningEditProposal(edit, new StringEditMotivation("Do not create cycles"));
                            
                            // Check orientation feasibility
                            if (!alreadyConsidered(edit, lastOrientationEdits) && !isBlocked(proposal)
                                    && (!onlyAllowedEdits || isOrientationAllowed(edit))) {
                                lastOrientationEdits.add(edit);
                                return proposal;
                            }
                        }
                        
                        if (!skip && nodeW.getChildren().contains(nodeY)) {
                            OrientLinkEdit edit = new OrientLinkEdit(probNet, nodeZ.getVariable(), nodeY.getVariable(), true);
                            LearningEditProposal proposal = new LearningEditProposal(edit, new StringEditMotivation("Do not create cycles"));
                            
                            // Check orientation feasibility
                            if (!alreadyConsidered(edit, lastOrientationEdits) && !isBlocked(proposal)
                                    && (!onlyAllowedEdits || isOrientationAllowed(edit))) {
                                lastOrientationEdits.add(edit);
                                return proposal;
                            }
                        }
                    }
                }
            }
        }
        return null;
    }
    
    /**
     * Attempts to orient non-directed links (X—Z) when no directed path exists in either direction,
     * as a last resort. Prioritizes orientations that preserve acyclicity and consistency.
     *
     * @param onlyAllowedEdits whether to restrict to allowed orientations
     * @return a LearningEditProposal if a safe orientation is found; null otherwise
     */
    private LearningEditProposal tryOrientUnorientedWithoutPath(boolean onlyAllowedEdits) {
        for (Link<Node> link : probNet.getLinks()) {
            if (!link.isDirected()) {
                Node nodeX = link.getNode1();
                Node nodeZ = link.getNode2();
                
                OrientLinkEdit edit = new OrientLinkEdit(probNet, nodeX.getVariable(), nodeZ.getVariable(), true);
                LearningEditProposal proposal = new LearningEditProposal(edit, new StringEditMotivation("Do not create cycles"));
                
                // Conditions:
                // 1. No path from Z to X to avoid cycle
                // 2. Not already considered
                // 3. Not blocked
                // 4. Allowed if filtering enabled
                if (!probNet.existsPath(nodeZ, nodeX, true) && !alreadyConsidered(edit, lastOrientationEdits)
                        && !isBlocked(proposal) && (!onlyAllowedEdits || isOrientationAllowed(edit))) {
                    lastOrientationEdits.add(edit);
                    return proposal;
                }
                
                // Try Z → X
                edit = new OrientLinkEdit(probNet, nodeZ.getVariable(), nodeX.getVariable(), true);
                proposal = new LearningEditProposal(edit, new StringEditMotivation("Do not create cycles"));
                
                // Same conditions, different direction
                if (!alreadyConsidered(edit, lastOrientationEdits) && !isBlocked(proposal)
                        && (!onlyAllowedEdits || isOrientationAllowed(edit))) {
                    lastOrientationEdits.add(edit);
                    return proposal;
                }
            }
        }
        return null;
    }
    
    /**
     * @param orientLinkEdit
     * @return true if the orientation is allowed, false otherwise
     */
    private boolean isOrientationAllowed(OrientLinkEdit orientLinkEdit) {
        Node sourceNode = probNet.getNode(orientLinkEdit.getVariable1());
        Node destinationNode = probNet.getNode(orientLinkEdit.getVariable2());
        return (
                !probNet.existsPath(destinationNode, sourceNode, true) && isAllowed(orientLinkEdit)
        );
    }
    
    @Override public void undoEditHappened(UndoableEditEvent event) {
        UndoableEdit edit = event.getEdit();
        Node nodeX, nodeY;
        
        if (edit instanceof RemoveLinkEdit removeLinkEdit) {
            phase = Phase.INITIAL_PHASE;
            nodeX = probNet.getNode(removeLinkEdit.getVariable1());
            nodeY = probNet.getNode(removeLinkEdit.getVariable2());
            List<Node> separationSet = cache.get(new NodePair(nodeX, nodeY)).getSeparationSet();
            double linkScore = independenceTester.test(caseDatabase, nodeX, nodeY, separationSet);
            cache.put(new NodePair(nodeX, nodeY), new PCEditMotivation(linkScore, separationSet));
        } else if (edit instanceof AddLinkEdit addLinkEdit) {
            nodeX = probNet.getNode(addLinkEdit.getVariable1());
            nodeY = probNet.getNode(addLinkEdit.getVariable2());
            probNet.removeLink(nodeX, nodeY, false);
            phase = Phase.INITIAL_PHASE;
        } else if (edit instanceof COrientLinksEdit) {
            phase = Phase.INITIAL_PHASE;
        } else if (edit instanceof OrientLinkEdit) {
            phase = Phase.HEAD_TO_HEAD_ORIENTATION;
        }
        resetHistory();
    }
    
    @Override public void undoableEditHappened(UndoableEditEvent event) {
        
        UndoableEdit edit = event.getEdit();
        Node nodeX, nodeY;
        
        if (edit instanceof RemoveLinkEdit removeLinkEdit) {
            nodeX = probNet.getNode(removeLinkEdit.getVariable1());
            nodeY = probNet.getNode(removeLinkEdit.getVariable2());

            PCEditMotivation cachedScore = cache.get(new NodePair(nodeX, nodeY));
            List<Node> separationSet = cachedScore != null ? cachedScore.getSeparationSet() : new ArrayList<>();
            cache.put(new NodePair(nodeX, nodeY), new PCEditMotivation(ALREADY_DONE, separationSet));

            // Remove the cached values X node's neighbors that contained Y in
            // the separation set (and vice versa)
            for (Node neighborNode : nodeX.getNeighbors()) {
                NodePair pair = new NodePair(nodeX, neighborNode);
                PCEditMotivation neighborScore = cache.get(pair);
                if (neighborScore != null && neighborScore.getScore() != ALREADY_DONE
                        && neighborScore.getSeparationSet().contains(nodeY)) {
                    cache.remove(pair);
                }
            }
            
        }
        //An AddLinkEdit can only be done by the user. Just undirect the link
        if (edit instanceof AddLinkEdit addLinkEdit) {
            nodeX = probNet.getNode(addLinkEdit.getVariable1());
            nodeY = probNet.getNode(addLinkEdit.getVariable2());
            probNet.removeLink(nodeX, nodeY, true);
            probNet.addLink(nodeX, nodeY, false);
            phase = Phase.INITIAL_PHASE;
        } else if (edit instanceof COrientLinksEdit) {
            // todo: check if this is correct
        }
        resetHistory();
    }
    
    /**
     * Returns the motivation of the edit. The motivation is a string
     */
    @Override public LearningEditMotivation getMotivation(PNEdit edit) {
        Node nodeX, nodeY, nodeZ;
        LearningEditMotivation motivation = null;
        if (edit instanceof RemoveLinkEdit removeLinkEdit) {
            nodeX = probNet.getNode(removeLinkEdit.getVariable1());
            nodeY = probNet.getNode(removeLinkEdit.getVariable2());
            motivation = cache.get(new NodePair(nodeX, nodeY));
            
        } else if (edit instanceof COrientLinksEdit compoundDirectLinkEdit) {
            nodeX = probNet.getNode(((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(0)).getVariable1());
            nodeZ = probNet.getNode(((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(0)).getVariable2());
            nodeY = probNet.getNode(((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(1)).getVariable1());
            motivation = new StringEditMotivation(
                    "Sep. set (" + nodeX.getName() + ", " + nodeY.getName() + ") does not contain variable: "
                            + nodeZ.getName());
        }
        if (edit instanceof OrientLinkEdit) {
            motivation = new StringEditMotivation("Do not create cycles");
        }
        return motivation;
    }
    
    @Override public boolean isLastPhase() {
        return (phase.ordinal() >= Phase.REMAINING_LINKS_ORIENTATION.ordinal());
    }
    
    /**
     * Clears the edits history: lastRemovedEdits, lastOrientationEdits, lastCompoundOrientationEdits
     */
    protected void resetHistory() {
        lastRemovedEdits.clear();
        lastOrientationEdits.clear();
        lastCompoundOrientationEdits.clear();
    }
    
}