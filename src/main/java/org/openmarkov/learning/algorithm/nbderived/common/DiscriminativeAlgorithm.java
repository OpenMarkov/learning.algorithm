package org.openmarkov.learning.algorithm.nbderived.common;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.learning.algorithm.naivebayes.IDiscriminativeBayes;
import org.openmarkov.learning.algorithm.scoreAndSearch.ScoreAndSearchAlgorithm;
import org.openmarkov.learning.core.util.LearningEditMotivation;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;
import org.openmarkov.learning.metric.Metric;

import java.util.List;
import java.util.stream.Collectors;


/**
 * Abstract base class for discriminative Bayesian classifier algorithms that extend Naive Bayes
 * with additional inter-feature links.
 * <p>
 * Subclasses that need a maximum weight spanning tree (e.g. TAN, FANB) should create and
 * manage a {@link MaximumWeightSpanningTree} instance directly.
 */
public abstract class DiscriminativeAlgorithm extends ScoreAndSearchAlgorithm implements IDiscriminativeBayes {

    /**
     * Manages the history of edits already considered in the current search cycle,
     * preventing the algorithm from re-proposing the same edit.
     */
    protected final EditHistorySupport editHistory = new EditHistorySupport();

    /**
     * Metric used to compute the Conditional Mutual Information for each pair of nodes conditioned to the class variable
     */
    protected Metric unconditionedMetric;



    /**
     * Constructs a discriminative algorithm with the given parameters.
     *
     * @param probNet      the probabilistic network to learn
     * @param caseDatabase the case database to learn from
     * @param metric       the scoring metric for evaluating inter-feature links
     * @param alpha        smoothing or significance parameter
     */
    public DiscriminativeAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Metric metric, Double alpha) {
        super(probNet, caseDatabase, metric, alpha);
    }


    /**
     * Marks the given edit as already considered in the current search cycle.
     *
     * @param edit the edit to mark
     */
    protected void markEditAsConsidered(BaseLinkEdit edit) {
        editHistory.markEditAsConsidered(edit);
    }

    /**
     * Checks whether the given edit has already been considered in the current search cycle.
     *
     * @param edit the edit to check
     * @return true if this edit was previously marked as considered
     */
    protected boolean isEditAlreadyConsidered(BaseLinkEdit edit) {
        return editHistory.isEditAlreadyConsidered(edit);
    }

    /**
     * Clears all recorded edit history, starting a fresh search cycle.
     */
    protected void resetHistory() {
        editHistory.reset();
    }


    @Override
    public LearningEditProposal getBestEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        resetHistory();
        return getNextEdit(onlyAllowedEdits, onlyPositiveEdits);
    }

    @Override
    public LearningEditProposal getNextEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        return getOptimalEdit(onlyAllowedEdits, onlyPositiveEdits);
    }

    @Override
    public LearningEditMotivation getMotivation(PNEdit edit) {
        return new ScoreEditMotivation(metric.getScore(edit));
    }

    /**
     * Subclass hook: returns the single best edit for the current search step.
     *
     * @param onlyAllowedEdits  if true, only constraint-satisfying edits are considered
     * @param onlyPositiveEdits if true, only edits with positive score are considered
     * @return the best edit proposal, or null if none is available
     */
    protected abstract LearningEditProposal getOptimalEdit(boolean onlyAllowedEdits,
                                                           boolean onlyPositiveEdits);


    /**
     * Sets the standard NB net given a root node
     */
    @Override public void setRelationsForRootVariable() {
        Node root = this.getRootNode();
        this.getNonRootNodes().forEach(node->{
            probNet.addLink(root, node, true);
            //markEditAsConsidered(new AddLinkEdit(probNet, root.getVariable(), node.getVariable(), true));
        });
    }



    @Override
    public Node getRootNode(){
        return this.probNet.getNodes().stream().filter(n-> n.getVariable().getName().equals(classVariableName)).findFirst().get();
    }

    @Override
    public List<Node> getNonRootNodes(){
        return  this.probNet.getNodes().stream().filter(n-> !n.getVariable().getName().equals(classVariableName)).collect(Collectors.toList());
    }




}
