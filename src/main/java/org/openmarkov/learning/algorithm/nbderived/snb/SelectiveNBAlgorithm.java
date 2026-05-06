package org.openmarkov.learning.algorithm.nbderived.snb;

import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.action.base.linkEdits.RemoveLinkEdit;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.constraint.MaxNumParents;
import org.openmarkov.core.model.network.constraint.NoCycle;
import org.openmarkov.learning.metric.Metric;
import org.openmarkov.learning.core.algorithm.LearningAlgorithmType;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ModelNetUse;
import org.openmarkov.learning.metric.cmi.accuracy.Accuracy;
import org.openmarkov.learning.algorithm.nbderived.common.DiscriminativeAlgorithm;

import java.util.Collection;
import java.util.stream.Collectors;


@LearningAlgorithmType(name = "Selective naive bayes", discriminative = true, supportsUnobservedVariables = false,
		metrics = "Accuracy")
public class SelectiveNBAlgorithm extends DiscriminativeAlgorithm {
    
    private double currentAccuracy = 0.0;

    private boolean forward;

    public SelectiveNBAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Metric metric, Double alpha) {
        super(probNet, caseDatabase, metric, alpha);
    }
    
    public SelectiveNBAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Metric metric, Double alpha, boolean fwd) {
        this(probNet, caseDatabase, metric, alpha);
        forward = fwd;
    }
    
    
    @Override public void init(ModelNetUse modelNetUse) {
        if (metric instanceof Accuracy) {
            ((Accuracy) metric).setClassVariable(this.classVariableName);
        }
        if (!forward) {
            setRelationsForRootVariable();
        }
        MaxNumParents maxNumParentsConstraint = new MaxNumParents(1);
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
        double bestPartialScore = !onlyAllowedEdits && !onlyPositiveEdits ? 0.0 : currentAccuracy;
        BaseLinkEdit bestEdit = null;
        LearningEditProposal bestEditProposal = null;
        ((Accuracy) metric).resetCache();

        Collection<Node> candidates = (!forward) ? getRootNode().getChildren() :
                getNonRootNodes().stream()
                                 .filter(n -> !getRootNode().getChildren().contains(n))
                                 .collect(Collectors.toSet());

        for (Node n1 : candidates) {
            BaseLinkEdit edit;
            if (forward) {
                edit = new AddLinkEdit(probNet, getRootNode().getVariable(), n1.getVariable(), true);
            } else {
                edit = new RemoveLinkEdit(probNet, getRootNode().getVariable(), n1.getVariable(), true);
            }

            double addScore = metric.getScore(edit);

            if (!isEditAlreadyConsidered(edit) && !isBlocked(edit)
                    && (!onlyAllowedEdits || isAllowed(edit))
                    && (addScore >= bestPartialScore || !onlyPositiveEdits)
            ) {
                bestEdit = edit;
                bestPartialScore = addScore;
            }
        }
        if (bestEdit != null) {
            bestEditProposal = LearningEditProposal.scored(bestEdit, bestPartialScore);
            markEditAsConsidered(bestEdit);
            currentAccuracy = bestPartialScore;
        }
        return bestEditProposal;
    }
}
