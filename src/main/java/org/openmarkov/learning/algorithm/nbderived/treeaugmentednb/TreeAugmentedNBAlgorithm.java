package org.openmarkov.learning.algorithm.nbderived.treeaugmentednb;

import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.constraint.MaxNumParents;
import org.openmarkov.core.model.network.constraint.NoCycle;
import org.openmarkov.learning.metric.Metric;
import org.openmarkov.learning.core.algorithm.LearningAlgorithmType;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ModelNetUse;
import org.openmarkov.learning.algorithm.nbderived.common.DiscriminativeAlgorithm;
import org.openmarkov.learning.algorithm.nbderived.common.MaximumWeightSpanningTree;
import org.openmarkov.learning.metric.cmi.mutualInformation.MutualInformationMetric;

import java.util.List;
import java.util.Random;


@LearningAlgorithmType(name = "Tree augmented naive bayes", discriminative = true, supportsUnobservedVariables = false,
		metrics = "ConditionalMutualInformation")
public class TreeAugmentedNBAlgorithm extends DiscriminativeAlgorithm {

    protected final MaximumWeightSpanningTree mwst = new MaximumWeightSpanningTree();

    public TreeAugmentedNBAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Metric metric, Double alpha) {
        super(probNet, caseDatabase, metric, alpha);
    }


    @Override public void init(ModelNetUse modelNetUse) {
        if (metric instanceof MutualInformationMetric) {
            ((MutualInformationMetric) metric).setClassVariable(this.classVariableName);
        }
        if (!mwst.isBuilt()) {
            mwst.build(probNet, metric, getNonRootNodes());
            Variable randomRoot = getRandomVariable();
            mwst.redirect(randomRoot, probNet);
        }
        setRelationsForRootVariable();

        MaxNumParents maxNumParentsConstraint = new MaxNumParents(2);
        this.probNet.addConstraint(new NoCycle());
        this.probNet.addConstraint(maxNumParentsConstraint);
    }

    /**
     * Selects a random non-root variable to use as the root for MWST redirect.
     * Overrideable for testing determinism.
     */
    protected Variable getRandomVariable() {
        return getNonRootVariables().get(new Random().nextInt(getNonRootVariables().size()));
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

        for (Node n1 : nodes) {
            for (Node n2 : nodes) {
                if (n2 == n1) {
                    continue;
                }
                AddLinkEdit addLink = new AddLinkEdit(probNet, n1.getVariable(), n2.getVariable(), true);
                double addScore = metric.getScore(addLink);

                if (!isEditAlreadyConsidered(addLink)
                        && ((!onlyAllowedEdits || isAllowed(addLink) && mwst.contains(n1.getVariable(), n2.getVariable()))
                        && (!onlyPositiveEdits || addScore > 0) && !isBlocked(addLink))
                ) {
                    bestEdit = addLink;
                    bestPartialScore = addScore;
                }
            }
        }
        if (bestEdit != null) {
            bestEditProposal = LearningEditProposal.scored(bestEdit, bestPartialScore);
            markEditAsConsidered(bestEdit);
        }
        return bestEditProposal;
    }
    
    
}
