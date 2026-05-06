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
import org.openmarkov.core.model.database.CaseDatabase;
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

import org.openmarkov.core.model.graph.Link;

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
 * <p>
 * Each phase is implemented by a dedicated class:
 * <ul>
 * <li>{@link SkeletonDiscovery} — skeleton discovery using PC-Stable</li>
 * <li>{@link ColliderOrientation} — v-structure (collider) detection and orientation</li>
 * <li>{@link MeekOrientation} — Meek rules R1/R2/R3 and fallback orientation</li>
 * </ul>
 *
 * @author Manuel Arias
 */
@LearningAlgorithmType(name = "PC", discriminative = false, supportsUnobservedVariables = false)
public class PCAlgorithm extends IndependenceRelationsAlgorithm
        implements PNEditListener {

    // Algorithm phases
    enum Phase {
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
    protected IndependenceTester independenceTester;

    /**
     * Degree of accuracy of the independence test.
     */
    protected double significanceLevel;

    /**
     * Optional causal direction tester for orienting remaining undirected links.
     * When non-null, it is used in the REMAINING_LINKS_ORIENTATION phase to prefer
     * the direction supported by the Additive Noise Model over an arbitrary choice.
     */
    CausalDirectionTester causalDirectionTester;

    /**
     * Current algorithm phase
     */
    private Phase phase;

    // Phase delegates
    private final SkeletonDiscovery skeleton;
    private final ColliderOrientation colliders;
    private final MeekOrientation meek;

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

        this.skeleton = new SkeletonDiscovery(this);
        this.colliders = new ColliderOrientation(this);
        this.meek = new MeekOrientation(this);
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
        skeleton.resetState();
        resetHistory();
    }

    // ---- Public API ----

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
     * For {@code INITIAL_PHASE}, delegates to {@link SkeletonDiscovery}.
     * For orientation phases, delegates to {@link ColliderOrientation} or
     * {@link MeekOrientation} as appropriate.
     *
     * @param onlyAllowedEdits  if true, only allowed edits are considered
     * @param onlyPositiveEdits if true, only positive edits are considered
     * @return LearningEditProposal
     */
    public LearningEditProposal getOptimalEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        if (phase == Phase.INITIAL_PHASE) {
            LearningEditProposal proposal = skeleton.findEdit(onlyAllowedEdits, onlyPositiveEdits);
            if (proposal != null) {
                return proposal;
            }
            return transitionToNextPhase(onlyAllowedEdits);
        }

        // Orientation phases: no depth-iteration needed; loop is just a safety bound.
        int adjacencySize = 0;
        LearningEditProposal bestEditProposal;
        while (maxOfAdjacencies() > adjacencySize) {
            bestEditProposal = findBestEditInCurrentPhase(onlyAllowedEdits);
            if (bestEditProposal != null) {
                return bestEditProposal;
            }
            adjacencySize++;
        }

        return transitionToNextPhase(onlyAllowedEdits);
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
        LearningEditProposal bestEdit = colliders.findEdit(onlyAllowedEdits);
        if (bestEdit == null) {
            if (!colliders.hasProducedEdits()) {
                phase = Phase.REMAINING_LINKS_ORIENTATION;
                LearningEditProposal p = meek.findEdit(onlyAllowedEdits);
                if (p == null && !meek.hasProducedEdits()) {
                    phase = Phase.ORIENTATION_FINISHED;
                }
                return p;
            }
        }
        return bestEdit;
    }

    /**
     * Returns the motivation of the edit.
     */
    @Override public LearningEditMotivation getMotivation(PNEdit edit) {
        Node nodeX, nodeY, nodeZ;
        LearningEditMotivation motivation = null;
        if (edit instanceof RemoveLinkEdit removeLinkEdit) {
            nodeX = probNet.getNode(removeLinkEdit.getVariableFrom());
            nodeY = probNet.getNode(removeLinkEdit.getVariableTo());
            motivation = cache.get(new NodePair(nodeX, nodeY));

        } else if (edit instanceof COrientLinksEdit compoundDirectLinkEdit) {
            nodeX = probNet.getNode(((OrientLinkEdit) compoundDirectLinkEdit.getEdits()
                                                                            .findFirst()
                                                                            .get()).getVariableFrom());
            nodeZ = probNet.getNode(((OrientLinkEdit) compoundDirectLinkEdit.getEdits()
                                                                            .findFirst()
                                                                            .get()).getVariableTo());
            nodeY = probNet.getNode(((OrientLinkEdit) compoundDirectLinkEdit.getEdits()
                                                                            .skip(1)
                                                                            .findFirst()
                                                                            .get()).getVariableFrom());
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

    // ---- PNEditListener callbacks ----
    // CRITICAL: CompoundEdit guard pattern.
    // When flattenEdit notifies sub-edits of a COrientLinksEdit, the order is:
    //   compound first, then sub-edits (OrientLinkEdits).
    // The guards below (phase != HEAD_TO_HEAD / phase != INITIAL_PHASE) prevent
    // sub-edit notifications from overriding the phase set by the compound handler.

    @Override public void afterEditExecutes(@UnknownNullability PNEdit edit) {
        if (edit instanceof RemoveLinkEdit removeLinkEdit) {
            skeleton.onEditExecuted(removeLinkEdit);
            phase = Phase.INITIAL_PHASE;
        }
        //An AddLinkEdit can only be done by the user. Just undirect the link
        if (edit instanceof AddLinkEdit addLinkEdit) {
            Node nodeX = probNet.getNode(addLinkEdit.getVariableFrom());
            Node nodeY = probNet.getNode(addLinkEdit.getVariableTo());
            probNet.removeLink(nodeX, nodeY, true);
            probNet.addLink(nodeX, nodeY, false);
            phase = Phase.INITIAL_PHASE;
            skeleton.resetState();
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

    @Override public void afterUndoingEdit(PNEdit edit) {
        if (edit instanceof RemoveLinkEdit removeLinkEdit) {
            phase = Phase.INITIAL_PHASE;
            skeleton.onEditUndone(removeLinkEdit);
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

    // ---- Phase dispatch and transitions ----

    /**
     * Dispatches orientation-phase best-edit search (HEAD_TO_HEAD / REMAINING).
     */
    private LearningEditProposal findBestEditInCurrentPhase(boolean onlyAllowedEdits) {
        return switch (phase) {
            case HEAD_TO_HEAD_ORIENTATION -> getOrientationEdit(onlyAllowedEdits);
            case REMAINING_LINKS_ORIENTATION -> {
                LearningEditProposal p = meek.findEdit(onlyAllowedEdits);
                if (p == null && !meek.hasProducedEdits()) {
                    phase = Phase.ORIENTATION_FINISHED;
                }
                yield p;
            }
            default -> null;
        };
    }

    /**
     * Transitions the algorithm to the next phase when no more edits are available
     * in the current one.
     */
    private LearningEditProposal transitionToNextPhase(boolean onlyAllowedEdits) {
        return switch (phase) {
            case INITIAL_PHASE -> {
                phase = Phase.HEAD_TO_HEAD_ORIENTATION;
                yield getOrientationEdit(onlyAllowedEdits);
            }
            case HEAD_TO_HEAD_ORIENTATION -> {
                phase = Phase.REMAINING_LINKS_ORIENTATION;
                LearningEditProposal p = meek.findEdit(onlyAllowedEdits);
                if (p == null && !meek.hasProducedEdits()) {
                    phase = Phase.ORIENTATION_FINISHED;
                }
                yield p;
            }
            case REMAINING_LINKS_ORIENTATION -> {
                phase = Phase.ORIENTATION_FINISHED;
                yield null;
            }
            case ORIENTATION_FINISHED -> null;
        };
    }

    /**
     * Clears the edits history across all phase delegates.
     */
    protected void resetHistory() {
        skeleton.resetHistory();
        colliders.resetHistory();
        meek.resetHistory();
    }

    // ---- Package-private accessors for phase classes ----
    // These are necessary because probNet, caseDatabase, isBlocked, and isAllowed
    // are protected in LearningAlgorithm (a different package), so they are not
    // accessible through a PCAlgorithm reference from non-subclasses in this package.

    ProbNet net() { return probNet; }
    CaseDatabase database() { return caseDatabase; }
    boolean checkBlocked(LearningEditProposal p) { return isBlocked(p); }
    boolean checkAllowed(PNEdit e) { return isAllowed(e); }

    /**
     * Returns the network's nodes sorted alphabetically by name.
     * Ensures deterministic iteration order regardless of insertion order.
     */
    List<Node> sortedNodes() {
        List<Node> nodes = new ArrayList<>(probNet.getNodes());
        nodes.sort(Comparator.comparing(Node::getName));
        return nodes;
    }

    /**
     * Returns a sorted copy of the given node list (alphabetical by name).
     */
    static List<Node> sorted(List<Node> nodes) {
        List<Node> copy = new ArrayList<>(nodes);
        copy.sort(Comparator.comparing(Node::getName));
        return copy;
    }

    /**
     * Returns the network's links sorted by (from-name, to-name).
     * Ensures deterministic iteration order for orientation phases.
     */
    List<Link<Node>> sortedLinks() {
        List<Link<Node>> links = new ArrayList<>(probNet.getLinks());
        links.sort(Comparator.comparing((Link<Node> l) -> l.getFrom().getName())
                .thenComparing(l -> l.getTo().getName()));
        return links;
    }

    /**
     * Checks whether an orientation is structurally allowed (no cycle creation, plus constraint check).
     */
    boolean isOrientationAllowed(OrientLinkEdit orientLinkEdit) {
        Node sourceNode = probNet.getNode(orientLinkEdit.getVariableFrom());
        Node destinationNode = probNet.getNode(orientLinkEdit.getVariableTo());
        return (
                !probNet.existsPath(destinationNode, sourceNode, true, Collections.emptyList()) && isAllowed(orientLinkEdit)
        );
    }

    // ---- Utilities ----

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
}
