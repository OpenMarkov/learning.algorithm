/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.pc;

import org.jetbrains.annotations.UnknownNullability;
import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.PNEditListener;
import org.openmarkov.core.action.core.COrientLinksEdit;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.graph.Link;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.action.base.linkEdits.OrientLinkEdit;
import org.openmarkov.core.action.base.linkEdits.RemoveLinkEdit;
import org.openmarkov.learning.algorithm.pc.independencetester.CausalDirectionTester;
import org.openmarkov.learning.algorithm.pc.independencetester.IndependenceTester;
import org.openmarkov.learning.core.algorithm.LearningAlgorithmType;
import org.openmarkov.learning.core.util.LearningEditMotivation;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ModelNetUse;
import org.openmarkov.learning.core.util.StringEditMotivation;
import org.openmarkov.learning.algorithm.pc.util.NodePair;

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
        implements PNEditListener {
    
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
     * Optional causal direction tester for orienting remaining undirected links.
     * When non-null, it is used in the REMAINING_LINKS_ORIENTATION phase to prefer
     * the direction supported by the Additive Noise Model over an arbitrary choice.
     */
    private final CausalDirectionTester causalDirectionTester;

    /**
     * Degree of accuracy of the independence test.
     */
    protected double significanceLevel;

    /**
     * PC-Stable: current conditioning-set depth being explored in the skeleton phase.
     * Persists across getBestEdit calls so all removals at depth d are found before
     * advancing to depth d+1 (order-independent skeleton discovery).
     */
    private int stableDepth = 0;

    /**
     * PC-Stable: snapshot of each node's neighbors at the start of {@link #stableDepth}.
     * Independence tests at depth d always use this frozen snapshot as the candidate
     * conditioning set, not the (potentially modified) live adjacency.
     * {@code null} means the snapshot has not been taken yet for the current depth.
     */
    private Map<Node, List<Node>> stableAdjSnapshot = null;

    /**
     * Current algorithm phase
     */
    private Phase phase;
    
    /**
     * Constructor for the PC Algorithm.
     *
     * @param probNet               Probabilistic Network to learn, initially it contains only the nodes.
     * @param caseDatabase          Database of cases
     * @param alpha                 Learning rate
     * @param independenceTester    Independence test method
     * @param significanceLevel     Statistical significance level
     * @param causalDirectionTester Optional tester for orienting remaining links; may be null
     */
    public PCAlgorithm(
            ProbNet probNet,
            CaseDatabase caseDatabase,
            Double alpha,
            IndependenceTester independenceTester,
            Double significanceLevel,
            CausalDirectionTester causalDirectionTester) {

        super(probNet, caseDatabase, alpha);
        this.independenceTester = independenceTester;
        this.significanceLevel = significanceLevel;
        this.causalDirectionTester = causalDirectionTester;
        this.probNet.getPNESupport().addListener(this);

        cache = new HashMap<>();

        this.phase = Phase.INITIAL_PHASE;
    }

    /**
     * Initializes the algorithm. Resets phase and cache so that a fresh run
     * (e.g. via the "Finish" button in the interactive dialog) is not affected
     * by phase pollution caused by the table-population peeking calls.
     */
    @Override
    public void init(ModelNetUse modelNetUse) {
        super.init(modelNetUse);
        phase = Phase.INITIAL_PHASE;
        cache.clear();
        resetSkeletonState();
        resetHistory();
    }

    /**
     * Method that returns the best edit in each step of the algorithm or null
     * if there are no more edits to consider.
     *
     * @param onlyAllowedEdits the only allowed edits
     * @param onlyPositiveEdits the only positive edits
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
     * Finds the optimal edit based on current algorithm phase.
     * <p>
     * For {@code INITIAL_PHASE}, uses PC-Stable logic: adjacency snapshots are frozen
     * at the start of each depth level so that all pairs at depth d are tested with the
     * same conditioning-set candidates, regardless of which edges have already been
     * removed at that depth.
     *
     * @param onlyAllowedEdits  if true, only allowed edits are considered
     * @param onlyPositiveEdits if true, only positive edits are considered
     * @return LearningEditProposal
     */
    public LearningEditProposal getOptimalEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        if (phase == Phase.INITIAL_PHASE) {
            return getOptimalEditInitialPhase(onlyAllowedEdits, onlyPositiveEdits);
        }

        // Orientation phases: no depth-iteration needed; loop is just a safety bound.
        int adjacencySize = 0;
        LearningEditProposal bestEditProposal;
        while (maxOfAdjacencies() > adjacencySize) {
            bestEditProposal = findBestEditInCurrentPhase(adjacencySize, onlyAllowedEdits, onlyPositiveEdits);
            if (bestEditProposal != null) {
                return bestEditProposal;
            }
            adjacencySize++;
        }

        return transitionToNextPhase(onlyAllowedEdits);
    }

    /**
     * PC-Stable skeleton discovery: iterates over increasing conditioning-set depths
     * using a frozen adjacency snapshot per depth, so that removals within a depth
     * do not affect the conditioning sets used for other pairs at the same depth.
     */
    private LearningEditProposal getOptimalEditInitialPhase(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        // Ensure a snapshot exists for the current depth.
        if (stableAdjSnapshot == null) {
            takeAdjacencySnapshot();
        }

        while (hasAnyPairAtDepth(stableDepth)) {
            separationSetsLogic(stableDepth);
            LearningEditProposal proposal = getOptimalEditFromCache(onlyAllowedEdits, onlyPositiveEdits);
            if (proposal != null) {
                return proposal;
            }
            // No removal found at this depth: advance and refresh snapshot.
            stableDepth++;
            takeAdjacencySnapshot();
        }

        return transitionToNextPhase(onlyAllowedEdits);
    }

    /**
     * Snapshots current adjacencies for use as PC-Stable conditioning-set candidates.
     */
    private void takeAdjacencySnapshot() {
        stableAdjSnapshot = new HashMap<>();
        for (Node node : probNet.getNodes()) {
            stableAdjSnapshot.put(node, new ArrayList<>(node.getNeighbors()));
        }
    }

    /**
     * Resets the PC-Stable skeleton state (depth counter and snapshot).
     * Must be called whenever the skeleton phase is restarted from scratch.
     */
    private void resetSkeletonState() {
        stableDepth = 0;
        stableAdjSnapshot = null;
    }

    /**
     * Returns true if there is at least one currently-present undirected edge (X–Y)
     * for which the snapshot conditioning set {@code snapshot[X] \ {Y}} has at least
     * {@code depth} elements and the edge has not yet been removed.
     */
    private boolean hasAnyPairAtDepth(int depth) {
        for (Node nodeX : probNet.getNodes()) {
            List<Node> snapshotNeighbors = stableAdjSnapshot.getOrDefault(nodeX, Collections.emptyList());
            // Need at least depth+1 snapshot neighbors so that snapshot[X]\{Y} has size >= depth
            if (snapshotNeighbors.size() - 1 < depth) {
                continue;
            }
            for (Node nodeY : nodeX.getSiblings()) {
                PCEditMotivation m = cache.get(new NodePair(nodeX, nodeY));
                if (m == null || m.getScore() != ALREADY_DONE) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Dispatches orientation-phase best-edit search (HEAD_TO_HEAD / REMAINING).
     * The {@code adjacencySize} parameter is kept for API symmetry but is not used
     * by either orientation method.
     */
    private LearningEditProposal findBestEditInCurrentPhase(int adjacencySize,
                                                            boolean onlyAllowedEdits,
                                                            boolean onlyPositiveEdits) {
        return switch (phase) {
            case HEAD_TO_HEAD_ORIENTATION -> getOrientationEdit(onlyAllowedEdits);
            case REMAINING_LINKS_ORIENTATION -> orientRemainingLinks(onlyAllowedEdits);
            default -> null;
        };
    }
    
    /**
     * Logic for evaluating separation sets in the INITIAL_PHASE.
     * Iterates through all currently-present undirected edges (X–Y) and tests
     * independence at the given depth.
     * <p>
     * <b>PC-Stable</b>: the candidate conditioning set for pair (X, Y) is taken from
     * {@link #stableAdjSnapshot}{@code [X] \ {Y}} — the snapshot captured at the
     * <em>start</em> of the current depth — rather than from the live adjacency of X.
     * This ensures that removing an edge at depth d does not alter the conditioning
     * sets used for other pairs at the same depth, making the skeleton order-independent.
     *
     * @param adjacencySize     Size of the conditioning set to consider (= stableDepth)
     * @param onlyPositiveEdits If true, only positive edits are considered
     */
    private void separationSetsLogic(int adjacencySize) {
        for (Node nodeX : probNet.getNodes()) {
            for (Node nodeY : nodeX.getSiblings()) {
                // PC-Stable: use the frozen snapshot for conditioning set candidates.
                List<Node> snapshotNeighbors = stableAdjSnapshot.getOrDefault(nodeX, Collections.emptyList());
                List<Node> adjacencySubset = new ArrayList<>(snapshotNeighbors);
                adjacencySubset.remove(nodeY);

                RemoveLinkEdit removeLinkEdit = new RemoveLinkEdit(
                        probNet, nodeX.getVariable(), nodeY.getVariable(), false);

                if (!alreadyConsidered(removeLinkEdit, lastRemovedEdits)) {
                	PCEditMotivation motivation = cache.get(new NodePair(nodeX, nodeY));

                    // Evaluate separation sets if not already cached or needs recalculation
                    if (motivation == null || (motivation.getScore() != ALREADY_DONE
                            && motivation.getSeparationSet().size() < adjacencySize)) {
                        evaluateSeparationSets(nodeX, nodeY, adjacencySubset, adjacencySize);
                    }
                }
            }
        }

    }

    /**
     * Transitions the algorithm to the next phase when no more edits are available
     * in the current one.
     *
     * @param onlyAllowedEdits if true, only structurally allowed edits are considered
     * @return the first proposal available in the next phase, or {@code null} if none exists
     */
    private LearningEditProposal transitionToNextPhase(boolean onlyAllowedEdits) {
        return switch (phase) {
            case INITIAL_PHASE -> {
                phase = Phase.HEAD_TO_HEAD_ORIENTATION;
                yield getOrientationEdit(onlyAllowedEdits);
            }
            case HEAD_TO_HEAD_ORIENTATION -> {
                phase = Phase.REMAINING_LINKS_ORIENTATION;
                yield orientRemainingLinks(onlyAllowedEdits);
            }
            case REMAINING_LINKS_ORIENTATION -> {
                phase = Phase.ORIENTATION_FINISHED;
                yield null;
            }
            case ORIENTATION_FINISHED -> null;
        };
    }
    
    /**
     * Evaluates separation sets for a given pair of nodes and updates the cache.
     * Always caches the best separation set found regardless of the significance level;
     * the onlyPositiveEdits filtering is applied later when displaying the table.
     *
     * @param nodeX the node x
     * @param nodeY the node y
     * @param adjacencySubset the adjacency subset
     * @param adjacencySize the adjacency size
     */
    private void evaluateSeparationSets(Node nodeX, Node nodeY, List<Node> adjacencySubset, int adjacencySize) {
        double bestScore = 0.0;
        List<Node> bestScoreSeparationSet = null;

        for (List<Node> separationSet : subSetsOfSize(adjacencySubset, adjacencySize)) {
            double linkScore = independenceTester.test(caseDatabase, nodeX, nodeY, separationSet);
            if (linkScore > bestScore) {
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
     * @param removeLinkEdit the remove link edit
     * @param bestMotivation the best motivation
     * @param onlyAllowedEdits the only allowed edits
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
     * @param onlyAllowedEdits the only allowed edits
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
        return new RemoveLinkEdit(probNet, edit.getVariableTo(), edit.getVariableFrom(), false);
    }
    
    /**
     * @param edit the edit
     * @param consideredEdits the considered edits
     * @return true if the edit has already been considered, false otherwise
     */
    public boolean alreadyConsidered(BaseLinkEdit edit, Set<PNEdit> consideredEdits) {
        BaseLinkEdit inverseEdit = new RemoveLinkEdit(probNet, edit.getVariableTo(), edit.getVariableFrom(),
                                                      edit.isDirected());
        return consideredEdits.contains(edit) || consideredEdits.contains(inverseEdit);
    }
    
    /**
     * @param edit1 the edit1
     * @param edit2 the edit2
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
                        ArrayList<OrientLinkEdit> edits = new ArrayList<>();

                        if (allowedXY && nodeX.isSibling(nodeY) && !createsContradictoryCollider(nodeX, nodeY)) {
                            // X–Y is undirected: orient X->Y
                            edits.add(orientXY);
                        }
                        if (allowedZY && nodeZ.isSibling(nodeY) && !createsContradictoryCollider(nodeZ, nodeY)) {
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
                                (edits.size() == 2) && alreadyConsidered(edits.get(0), edits.get(1));

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
     * Returns true if orienting {@code from → to} would create an unshielded collider
     * {@code from → to ← existingParent} that contradicts the separation set of the pair
     * {@code (from, existingParent)}.
     *
     * <p>A contradiction occurs when {@code to} belongs to {@code sep(from, existingParent)},
     * meaning the original skeleton phase concluded that conditioning on {@code to} makes
     * {@code from} and {@code existingParent} independent — i.e., {@code to} is a non-collider
     * on that path.  Orienting {@code from → to} would make it a collider, which is inconsistent.
     *
     * @param from the proposed tail node of the new directed edge
     * @param to   the proposed head node of the new directed edge
     * @return true if the orientation would produce a contradictory collider
     */
    private boolean createsContradictoryCollider(Node from, Node to) {
        for (Node existingParent : to.getParents()) {
            // Only unshielded triples matter (from and existingParent must not be adjacent)
            if (from.getNeighbors().contains(existingParent)) {
                continue;
            }
            PCEditMotivation sep = cache.get(new NodePair(from, existingParent));
            if (sep != null && sep.getSeparationSet().contains(to)) {
                return true;
            }
            // Also check the reverse key order
            sep = cache.get(new NodePair(existingParent, from));
            if (sep != null && sep.getSeparationSet().contains(to)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Orients remaining undirected links using Meek's orientation rules (R1, R2, R3)
     * followed by a fallback for truly undetermined edges.
     * <p>
     * Meek (1995) proves that these rules, applied to closure, produce the unique CPDAG
     * (completed partially directed acyclic graph) representing the Markov equivalence
     * class of the true DAG.
     */
    private LearningEditProposal orientRemainingLinks(boolean onlyAllowedEdits) {
        // R1: A→B—C, A⊥C  →  B→C   (avoid new unshielded collider A→B←C)
        LearningEditProposal p = meekR1(onlyAllowedEdits);
        if (p != null) return p;

        // R2: A→B→C, A—C  →  A→C   (avoid directed cycle A→B→C→A)
        p = meekR2(onlyAllowedEdits);
        if (p != null) return p;

        // R3: D—A, D—B, D—C, B→A, C→A, B⊥C  →  D→A
        //     (avoid new unshielded collider B→A←C created if A→D were chosen,
        //      which would make B→A→D←C an unshielded collider at D since B⊥C)
        p = meekR3(onlyAllowedEdits);
        if (p != null) return p;

        // Fallback: orient using ANM if available, else arbitrary (preserves acyclicity)
        p = tryOrientUnorientedWithoutPath(onlyAllowedEdits);
        if (p != null) return p;

        if (lastOrientationEdits.isEmpty()) {
            phase = Phase.ORIENTATION_FINISHED;
        }
        return null;
    }

    /**
     * Meek Rule R1: for each undirected edge B—C, if there exists A→B where A and C
     * are not adjacent, orient B→C.
     * <p>
     * Rationale: orienting C→B instead would create a new unshielded collider A→B←C
     * (unshielded because A⊥C), which contradicts the collider set fixed in phase 2.
     */
    private LearningEditProposal meekR1(boolean onlyAllowedEdits) {
        for (Link<Node> link : probNet.getLinks()) {
            if (!link.isDirected()) {
                LearningEditProposal p = meekR1Orient(link.getFrom(), link.getTo(), onlyAllowedEdits);
                if (p != null) return p;
                p = meekR1Orient(link.getTo(), link.getFrom(), onlyAllowedEdits);
                if (p != null) return p;
            }
        }
        return null;
    }

    /** Tries to apply R1 orienting nodeB→nodeC, given an existing parent A of nodeB. */
    private LearningEditProposal meekR1Orient(Node nodeB, Node nodeC, boolean onlyAllowedEdits) {
        for (Node nodeA : nodeB.getParents()) {
            if (!nodeA.getNeighbors().contains(nodeC)) {
                LearningEditProposal proposal = buildOrientProposal(nodeB, nodeC, "Meek R1", onlyAllowedEdits);
                if (proposal != null) return proposal;
            }
        }
        return null;
    }

    /**
     * Meek Rule R2: for each undirected edge A—C, if there exists a directed path A⟹C,
     * orient A→C.
     * <p>
     * Rationale: orienting C→A instead would create a directed cycle.
     * The original R2 states the pattern A→B→C; checking for any directed path A⟹C
     * is equivalent and handles longer chains uniformly.
     */
    private LearningEditProposal meekR2(boolean onlyAllowedEdits) {
        for (Link<Node> link : probNet.getLinks()) {
            if (!link.isDirected()) {
                LearningEditProposal p = meekR2Orient(link.getFrom(), link.getTo(), onlyAllowedEdits);
                if (p != null) return p;
                p = meekR2Orient(link.getTo(), link.getFrom(), onlyAllowedEdits);
                if (p != null) return p;
            }
        }
        return null;
    }

    /** Tries to apply R2 orienting nodeA→nodeC when a directed path nodeA⟹nodeC exists. */
    private LearningEditProposal meekR2Orient(Node nodeA, Node nodeC, boolean onlyAllowedEdits) {
        if (probNet.existsPath(nodeA, nodeC, true, Collections.emptyList())) {
            return buildOrientProposal(nodeA, nodeC, "Meek R2", onlyAllowedEdits);
        }
        return null;
    }

    /**
     * Meek Rule R3: for each undirected edge D—A, if there exist two distinct nodes B and C
     * such that D—B (undirected), D—C (undirected), B→A, C→A, and B not adjacent to C,
     * orient D→A.
     * <p>
     * Rationale: orienting A→D instead would create a new unshielded collider at D
     * in the path B→A→D←C, since B and C are not adjacent.
     */
    private LearningEditProposal meekR3(boolean onlyAllowedEdits) {
        for (Link<Node> link : probNet.getLinks()) {
            if (!link.isDirected()) {
                LearningEditProposal p = meekR3Orient(link.getFrom(), link.getTo(), onlyAllowedEdits);
                if (p != null) return p;
                p = meekR3Orient(link.getTo(), link.getFrom(), onlyAllowedEdits);
                if (p != null) return p;
            }
        }
        return null;
    }

    /**
     * Tries to apply R3 orienting nodeD→nodeA.
     * Looks for a pair (B, C) that are both parents of nodeA, both undirected siblings
     * of nodeD, and not adjacent to each other.
     */
    private LearningEditProposal meekR3Orient(Node nodeD, Node nodeA, boolean onlyAllowedEdits) {
        // Candidates: parents of A that are also undirected siblings of D
        List<Node> candidates = new ArrayList<>(nodeA.getParents());
        candidates.retainAll(nodeD.getSiblings());

        for (int i = 0; i < candidates.size(); i++) {
            Node nodeB = candidates.get(i);
            for (int j = i + 1; j < candidates.size(); j++) {
                Node nodeC = candidates.get(j);
                if (!nodeB.getNeighbors().contains(nodeC)) {
                    LearningEditProposal proposal = buildOrientProposal(nodeD, nodeA, "Meek R3", onlyAllowedEdits);
                    if (proposal != null) return proposal;
                }
            }
        }
        return null;
    }

    /**
     * Builds and validates an {@link OrientLinkEdit} proposal for orienting {@code from→to}.
     * Returns {@code null} if the edit was already considered, is blocked, or is not allowed.
     */
    private LearningEditProposal buildOrientProposal(Node from, Node to, String motivation, boolean onlyAllowedEdits) {
        OrientLinkEdit edit = new OrientLinkEdit(probNet, from.getVariable(), to.getVariable(), true);
        LearningEditProposal proposal = new LearningEditProposal(edit, new StringEditMotivation(motivation));
        if (!alreadyConsidered(edit, lastOrientationEdits)
                && !isBlocked(proposal)
                && (!onlyAllowedEdits || isOrientationAllowed(edit))) {
            lastOrientationEdits.add(edit);
            return proposal;
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
                Node nodeX = link.getFrom();
                Node nodeZ = link.getTo();

                // When an ANM tester is available, use it to determine the preferred causal direction.
                // The tester returns a p-value: higher means stronger evidence for that direction.
                // We try the preferred direction first and fall back to the other if it would create a cycle.
                Node preferred = nodeX;
                Node other    = nodeZ;
                String motivation = "Do not create cycles";
                if (causalDirectionTester != null) {
                    double scoreXZ = causalDirectionTester.testDirection(caseDatabase, nodeX, nodeZ);
                    double scoreZX = causalDirectionTester.testDirection(caseDatabase, nodeZ, nodeX);
                    if (scoreZX > scoreXZ) {
                        preferred = nodeZ;
                        other     = nodeX;
                    }
                    motivation = "Causal direction test (ANM)";
                }

                // Try preferred direction first
                OrientLinkEdit edit = new OrientLinkEdit(probNet, preferred.getVariable(), other.getVariable(), true);
                LearningEditProposal proposal = new LearningEditProposal(edit, new StringEditMotivation(motivation));
                if (!probNet.existsPath(other, preferred, true, Collections.emptyList())
                        && !alreadyConsidered(edit, lastOrientationEdits)
                        && !isBlocked(proposal)
                        && (!onlyAllowedEdits || isOrientationAllowed(edit))) {
                    lastOrientationEdits.add(edit);
                    return proposal;
                }

                // Fall back to opposite direction
                edit = new OrientLinkEdit(probNet, other.getVariable(), preferred.getVariable(), true);
                proposal = new LearningEditProposal(edit, new StringEditMotivation("Do not create cycles"));
                if (!probNet.existsPath(preferred, other, true, Collections.emptyList())
                        && !alreadyConsidered(edit, lastOrientationEdits)
                        && !isBlocked(proposal)
                        && (!onlyAllowedEdits || isOrientationAllowed(edit))) {
                    lastOrientationEdits.add(edit);
                    return proposal;
                }
            }
        }
        return null;
    }
    
    /**
     * @param orientLinkEdit the orient link edit
     * @return true if the orientation is allowed, false otherwise
     */
    private boolean isOrientationAllowed(OrientLinkEdit orientLinkEdit) {
        Node sourceNode = probNet.getNode(orientLinkEdit.getVariableFrom());
        Node destinationNode = probNet.getNode(orientLinkEdit.getVariableTo());
        return (
                !probNet.existsPath(destinationNode, sourceNode, true, Collections.emptyList()) && isAllowed(orientLinkEdit)
        );
    }
    
    @Override public void afterUndoingEdit(PNEdit edit) {
        if (edit instanceof RemoveLinkEdit removeLinkEdit) {
            phase = Phase.INITIAL_PHASE;
            resetSkeletonState();
            Node nodeX = probNet.getNode(removeLinkEdit.getVariableFrom());
            Node nodeY = probNet.getNode(removeLinkEdit.getVariableTo());
            List<Node> separationSet = cache.get(new NodePair(nodeX, nodeY)).getSeparationSet();
            double linkScore = independenceTester.test(caseDatabase, nodeX, nodeY, separationSet);
            cache.put(new NodePair(nodeX, nodeY), new PCEditMotivation(linkScore, separationSet));
        } else if (edit instanceof AddLinkEdit addLinkEdit) {
            Node nodeX = probNet.getNode(addLinkEdit.getVariableFrom());
            Node nodeY = probNet.getNode(addLinkEdit.getVariableTo());
            probNet.removeLink(nodeX, nodeY, false);
            phase = Phase.INITIAL_PHASE;
        } else if (edit instanceof COrientLinksEdit) {
            phase = Phase.INITIAL_PHASE;
        } else if (edit instanceof OrientLinkEdit) {
            // Guard: when flattenEdit notifies sub-edits of a COrientLinksEdit, the compound
            // handler above fires first and sets INITIAL_PHASE.  Sub-edits must not override that.
            if (phase != Phase.INITIAL_PHASE) {
                phase = Phase.HEAD_TO_HEAD_ORIENTATION;
            }
        }
        resetHistory();
    }

    @Override public void afterEditExecutes(@UnknownNullability PNEdit edit) {
        if (edit instanceof RemoveLinkEdit removeLinkEdit) {
            Node nodeX = probNet.getNode(removeLinkEdit.getVariableFrom());
            Node nodeY = probNet.getNode(removeLinkEdit.getVariableTo());

            PCEditMotivation cachedScore = cache.get(new NodePair(nodeX, nodeY));
            List<Node> separationSet = cachedScore != null ? cachedScore.getSeparationSet() : new ArrayList<>();
            cache.put(new NodePair(nodeX, nodeY), new PCEditMotivation(ALREADY_DONE, separationSet));

            // Invalidate cached tests whose separation set contained X or Y,
            // since those conditioning sets may no longer be subsets of the new adjacency.
            // Both sides must be invalidated (symmetric).
            for (Node neighborNode : nodeX.getNeighbors()) {
                NodePair pair = new NodePair(nodeX, neighborNode);
                PCEditMotivation neighborScore = cache.get(pair);
                if (neighborScore != null && neighborScore.getScore() != ALREADY_DONE
                        && neighborScore.getSeparationSet().contains(nodeY)) {
                    cache.remove(pair);
                }
            }
            for (Node neighborNode : nodeY.getNeighbors()) {
                NodePair pair = new NodePair(nodeY, neighborNode);
                PCEditMotivation neighborScore = cache.get(pair);
                if (neighborScore != null && neighborScore.getScore() != ALREADY_DONE
                        && neighborScore.getSeparationSet().contains(nodeX)) {
                    cache.remove(pair);
                }
            }

            // Reset phase and PC-Stable skeleton state: getNextEdit() called for
            // table population in interactive mode may have advanced stableDepth
            // well beyond what the current graph needs (by peeking through all depths
            // until null is returned).  If we only reset phase but not stableDepth,
            // the next call to getOptimalEditInitialPhase() will find
            // hasAnyPairAtDepth(stableDepth)==false immediately (stale snapshot, high
            // depth) and jump straight to orientation, skipping pending removals.
            phase = Phase.INITIAL_PHASE;
            resetSkeletonState();
        }
        //An AddLinkEdit can only be done by the user. Just undirect the link
        if (edit instanceof AddLinkEdit addLinkEdit) {
            Node nodeX = probNet.getNode(addLinkEdit.getVariableFrom());
            Node nodeY = probNet.getNode(addLinkEdit.getVariableTo());
            probNet.removeLink(nodeX, nodeY, true);
            probNet.addLink(nodeX, nodeY, false);
            phase = Phase.INITIAL_PHASE;
            resetSkeletonState();
        } else if (edit instanceof COrientLinksEdit) {
            // After applying a v-structure orientation, reset to HEAD_TO_HEAD_ORIENTATION.
            // During interactive table population, getNextEdit() peeks ahead and may advance
            // the phase all the way to ORIENTATION_FINISHED via transitionToNextPhase().
            // Without this reset, the phase would remain ORIENTATION_FINISHED after the user
            // accepts the edit, causing getBestEdit() to return null and the list to appear empty.
            phase = Phase.HEAD_TO_HEAD_ORIENTATION;
        } else if (edit instanceof OrientLinkEdit) {
            // After applying a remaining-link orientation, reset to REMAINING_LINKS_ORIENTATION.
            // Same peeking issue can advance the phase to ORIENTATION_FINISHED prematurely.
            //
            // Guard: when flattenEdit notifies sub-edits of a COrientLinksEdit, the compound
            // handler above fires first and sets HEAD_TO_HEAD_ORIENTATION.  The sub-edits
            // (OrientLinkEdits) must NOT override that — they should only act on standalone
            // orient edits from the remaining-links phase.
            if (phase != Phase.HEAD_TO_HEAD_ORIENTATION) {
                phase = Phase.REMAINING_LINKS_ORIENTATION;
            }
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
            nodeX = probNet.getNode(removeLinkEdit.getVariableFrom());
            nodeY = probNet.getNode(removeLinkEdit.getVariableTo());
            motivation = cache.get(new NodePair(nodeX, nodeY));
            
        } else if (edit instanceof COrientLinksEdit compoundDirectLinkEdit) {
            nodeX = probNet.getNode(((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(0)).getVariableFrom());
            nodeZ = probNet.getNode(((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(0)).getVariableTo());
            nodeY = probNet.getNode(((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(1)).getVariableFrom());
            motivation = new StringEditMotivation(
                    "Sep. set (" + nodeX.getName() + ", " + nodeY.getName() + ") does not contain variable: "
                            + nodeZ.getName());
        }
        if (edit instanceof OrientLinkEdit) {
            motivation = new StringEditMotivation("Meek orientation rule");
        }
        return motivation;
    }
    
    @Override public int getPhase() {
        return phase.ordinal();
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