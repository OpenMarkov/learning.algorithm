package org.openmarkov.learning.algorithm.nbderived.treeaugmentednb;

import org.openmarkov.core.action.AddLinkEdit;
import org.openmarkov.core.action.BaseLinkEdit;
import org.openmarkov.core.action.PNEdit;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.constraint.MaxNumParents;
import org.openmarkov.core.model.network.constraint.NoCycle;
import org.openmarkov.learning.metric.Metric;
import org.openmarkov.learning.core.algorithm.LearningAlgorithmType;
import org.openmarkov.learning.core.util.LearningEditMotivation;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ModelNetUse;
import org.openmarkov.learning.core.util.ScoreEditMotivation;
import org.openmarkov.learning.algorithm.nbderived.common.DiscriminativeAlgorithm;
import org.openmarkov.learning.algorithm.nbderived.treeaugmentednb.util.TreeAugmentedNBEditProposal;
import org.openmarkov.learning.metric.cmi.mutualInformation.MutualInformationMetric;

import java.util.ArrayList;
import java.util.List;


@LearningAlgorithmType(name = "Tree augmented naive bayes", discriminative = true, supportsUnobservedVariables = false)
public class TreeAugmentedNBAlgorithm extends DiscriminativeAlgorithm {
    
    /**
     * List with the best edits that have not been done by the
     * algorithm because they violate the ModelNetworkConstraint
     */
    protected List<PNEdit> lastBestEdits;
    
    public TreeAugmentedNBAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Metric metric, Double alpha) {
        super(probNet, caseDatabase, metric, alpha);
        this.lastBestEdits = new ArrayList<PNEdit>();
    }
    
    
    /**
     * This method returns the best edit (and its associated score)
     * that can be done to the network that is being learnt.
     *
     * @param onlyAllowedEdits  If this parameter is true, only those edits
     *                          that do not provoke a ConstraintViolated are returned
     * @param onlyPositiveEdits If this parameter is true, only those
     *                          edits with a positive associated score are returned.
     * @return {@code LearningEditProposal} with the best edit and its score.
     */
    @Override public LearningEditProposal getBestEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        resetHistory();
        return getNextEdit(onlyAllowedEdits, onlyPositiveEdits);
    }
    
    /**
     * Store last best edit
     *
     * @param edit
     */
    protected void markEditAsConsidered(BaseLinkEdit edit) {
        this.lastBestEdits.add(edit);
    }
    
    /**
     * @param edit
     * @return true if it is an edit already considered
     */
    protected boolean isEditAlreadyConsidered(BaseLinkEdit edit) {
        return this.lastBestEdits.contains(edit);
    }
    
    
    protected void resetHistory() {
        lastBestEdits.clear();
    }
    
    
    @Override public LearningEditMotivation getMotivation(PNEdit edit) {
        return new ScoreEditMotivation(metric.getScore(edit));
    }
    
    
    /**
     * This method returns the next best edit (and its associated score)
     * that can be done to the network that is being learnt.
     *
     * @param onlyAllowedEdits  If this parameter is true, only those edits
     *                          that do not provoke a ConstraintViolated are returned
     * @param onlyPositiveEdits If this parameter is true, only those
     *                          edits with a positive associated score are returned.
     * @return {@code LearningEditProposal} with the best edit and its score.
     */
    @Override public LearningEditProposal getNextEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        return getOptimalEdit(probNet, onlyAllowedEdits, onlyPositiveEdits);
    }
    
    @Override public void init(ModelNetUse modelNetUse) {
        if (metric instanceof MutualInformationMetric) {
            ((MutualInformationMetric) metric).setClassVariable(this.classVariableName);
        }
        if (maximumWeightSpanningTree.isEmpty()) {
            buildMaximumWeightSpanningTree();
            directedMaxWeightSpanningTree = redirectMaximumWeightSpanningTree(getRandomVariable());
        }
        setRelationsForRootVariable();
        MaxNumParents maxNumParentsConstraint = new MaxNumParents();
        maxNumParentsConstraint.setMaxNumParents(2);
        this.probNet.addConstraint(new NoCycle());
        this.probNet.addConstraint(maxNumParentsConstraint);
    }
    
    
    /**
     * Method to obtain the edit with the highest associated score.
     *
     * @param learnedNet net to learn.
     * @return {@code PNEdit} edit with the highest associated score.
     */
    private LearningEditProposal getOptimalEdit(ProbNet learnedNet, boolean onlyAllowedEdits,
                                                boolean onlyPositiveEdits) {
        final double[] bestPartialScore = {Double.NEGATIVE_INFINITY};
        final BaseLinkEdit[] bestEdit = {null};
        LearningEditProposal bestEditProposal = null;
        List<Node> nodes = getNonRootNodes();
        
        nodes.forEach(n1 -> {
            nodes.stream().filter(n -> n != n1).forEach(n2 -> {
                AddLinkEdit addLink = new AddLinkEdit(learnedNet, n1.getVariable(), n2.getVariable(), true);
                double addScore = metric.getScore(addLink);
                
                if (!isEditAlreadyConsidered(addLink) //&& addScore >= bestPartialScore[0]
                        && ((!onlyAllowedEdits || isAllowed(addLink) && withinMaxWeightSpanningTree(n1.getVariable(), n2.getVariable()))
                        && (!onlyPositiveEdits || addScore > 0) && !isBlocked(addLink))
                ) {
                    bestEdit[0] = addLink;
                    bestPartialScore[0] = addScore;
                }
            });
        });
        if (bestEdit[0] != null) {
            bestEditProposal = new TreeAugmentedNBEditProposal(bestEdit[0], bestPartialScore[0]);
            markEditAsConsidered(bestEdit[0]);
        }
        return bestEditProposal;
    }
    
    
}
