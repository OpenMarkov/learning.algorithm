package org.openmarkov.learning.algorithm.nbderived.fanb;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.model.database.CaseDatabase;
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
import org.openmarkov.learning.algorithm.nbderived.common.MaximumWeightSpanningTree;
import org.openmarkov.learning.metric.cmi.mutualInformation.MutualInformationMetric;

import java.util.List;


@LearningAlgorithmType(name = "Forest augmented naive bayes", discriminative = true, supportsUnobservedVariables = false,
		metrics = {"ConditionalMutualInformation", "MutualInformation"})
public class ForestAugmentedNBAlgorithm extends DiscriminativeAlgorithm {

    protected final MaximumWeightSpanningTree mwst = new MaximumWeightSpanningTree();

    /**
     * Maximum allowable degree of feature dependence
     */
    protected int kDependence;

    /**
     * Threshold to filter class conditioned links between nodes
     */
    protected Double avgCMI;

    /**
     * Node that would be used to re-direct the links in the maximum weight spanning tree
     */
    protected Node subtreeRoot;
    
    
    public ForestAugmentedNBAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Metric metric, Metric unconditioned, Double alpha) {
        super(probNet, caseDatabase, metric, alpha);
        this.unconditionedMetric = unconditioned;
        this.unconditionedMetric.init(probNet, caseDatabase);
    }
    
    
    @Override public LearningEditMotivation getMotivation(PNEdit edit) {
        return new ScoreEditMotivation(
                (((BaseLinkEdit) edit).getVariableFrom().getName().equals(getRootNode().getName()) ?
                        unconditionedMetric : metric).getScore(edit)
        );
    }

    @Override public void init(ModelNetUse modelNetUse) {
        kDependence = 1;
        if (metric instanceof MutualInformationMetric miMetric) {
            miMetric.setClassVariable(this.classVariableName);
        }
        if (unconditionedMetric instanceof MutualInformationMetric miMetric) {
            miMetric.setClassVariable(this.classVariableName);
        }
        if (!mwst.isBuilt()) {
            mwst.build(probNet, metric, getNonRootNodes());
        }
        avgCMI = avgCMI == null ? computeAveragedConditionalMutualInformation() : avgCMI;
        setRelationsForRootVariable();
        MaxNumParents maxNumParentsConstraint = new MaxNumParents(kDependence + 1);
        this.probNet.addConstraint(new NoCycle());
        this.probNet.addConstraint(maxNumParentsConstraint);
    }
    
    
    /**
     * Method to obtain the edit with the highest associated score.
     *
     * @param learnedNet net to learn.
     * @return {@code PNEdit} edit with the highest associated score.
     */
    @Override
    protected LearningEditProposal getOptimalEdit(boolean onlyAllowedEdits,
                                                  boolean onlyPositiveEdits) {
        double bestPartialScore = Double.NEGATIVE_INFINITY;
        BaseLinkEdit bestEdit = null;
        LearningEditProposal bestEditProposal = null;
        List<Node> nodes = getNonRootNodes();

        if (subtreeRoot == null) {
            bestEdit = getBestRootForSubtree();
            subtreeRoot = probNet.getNode(bestEdit.getVariableTo());
            mwst.redirect(subtreeRoot.getVariable(), probNet);
            bestPartialScore = unconditionedMetric.getScore(bestEdit);
        } else {
            for (Node n1 : nodes) {
                for (Node n2 : nodes) {
                    if (n2 == n1 || n2 == subtreeRoot) {
                        continue;
                    }
                    AddLinkEdit addLink = new AddLinkEdit(probNet, n1.getVariable(), n2.getVariable(), true);
                    double addScore = metric.getScore(addLink);

                    if ((addScore >= bestPartialScore) && (addScore > avgCMI) && !isEditAlreadyConsidered(addLink)
                            && ((!onlyAllowedEdits || isAllowed(addLink) && mwst.contains(n1.getVariable(), n2.getVariable()))
                            && (!onlyPositiveEdits || addScore > 0) && !isBlocked(addLink))
                    ) {
                        bestEdit = addLink;
                        bestPartialScore = addScore;
                    }
                }
            }
        }

        if (bestEdit != null) {
            bestEditProposal = LearningEditProposal.scored(bestEdit, bestPartialScore);
            markEditAsConsidered(bestEdit);
        }
        return bestEditProposal;
    }
    
    
    /**
     * Retrieves all the conditional mutual information values and averages them based on the number of nodes
     *
     * @return averaged conditional mutual information
     */
    protected Double computeAveragedConditionalMutualInformation() {
        double score = 0.0;
        List<Node> nonRootNodes = getNonRootNodes();
        int nodesSize = nonRootNodes.size();
        for (Node n1 : nonRootNodes) {
            for (Node n2 : nonRootNodes) {
                score += metric.getScore(new AddLinkEdit(probNet, n1.getVariable(), n2.getVariable(), true));
            }
        }

        return (score / (nodesSize * (nodesSize - 1)));
    }
    
    
    protected BaseLinkEdit getBestRootForSubtree() {
        BaseLinkEdit bestEdit = null;
        double bestScore = 0.0;
        for (Node n1 : getNonRootNodes()) {
            BaseLinkEdit ble = new AddLinkEdit(probNet, getRootNode().getVariable(), n1.getVariable(), true);
            double addScore = unconditionedMetric.getScore(ble);
            if (addScore > bestScore) {
                bestEdit = ble;
                bestScore = addScore;
            }
        }
        return bestEdit;
    }
    
}