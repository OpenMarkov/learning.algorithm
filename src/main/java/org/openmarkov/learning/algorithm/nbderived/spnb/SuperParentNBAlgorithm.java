package org.openmarkov.learning.algorithm.nbderived.spnb;

import org.openmarkov.core.action.base.ListPNEdit;
import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.constraint.DistinctLinks;
import org.openmarkov.core.model.network.constraint.MaxNumParents;
import org.openmarkov.learning.metric.Metric;
import org.openmarkov.learning.core.algorithm.LearningAlgorithmType;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ModelNetUse;
import org.openmarkov.learning.core.util.ScoreEditMotivation;
import org.openmarkov.learning.metric.cmi.accuracy.Accuracy;
import org.openmarkov.learning.algorithm.nbderived.common.DiscriminativeAlgorithm;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;


@LearningAlgorithmType(name = "Superparent naive bayes", discriminative = true, supportsUnobservedVariables = false,
		metrics = "Accuracy")
public class SuperParentNBAlgorithm extends DiscriminativeAlgorithm {
    
    
    /**
     *         SUPERPARENT ALGORITHM
     *             0. Initialize network to naive Bayes.
     *             1. Evaluate the current classifier.
     *             2. Consider making each node a SuperParent.  Let ASP be the SuperParent which increases accuracy the most.
     *             3. Consider an arc from ASP to each orphan. If the best such arc improves accuracy, keep it and go to 2.
     *             Else: Return the current classifier.
     */
    
    
    private LinkedList<Node> superParents;
    
    private LinkedList<Node> orphans;
    
    private boolean sameSP;
    
    private double currentAccuracy = 0.0;
    
    public SuperParentNBAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Metric metric, Double alpha) {
        super(probNet, caseDatabase, metric, alpha);
    }
    
    public SuperParentNBAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Metric metric, Double alpha, boolean sp) {
        this(probNet, caseDatabase, metric, alpha);
        this.sameSP = sp;
    }
    
    
    @Override public void init(ModelNetUse modelNetUse) {
        if (metric instanceof Accuracy) {
            ((Accuracy) metric).setClassVariable(this.classVariableName);
            ((Accuracy) metric).setAugmentedNet(true);
            ((Accuracy) metric).setAlpha(alpha);
            
        }
        superParents = new LinkedList<>();
        orphans = new LinkedList<Node>(getNonRootNodes());
        
        setRelationsForRootVariable();
        MaxNumParents maxNumParentsConstraint = new MaxNumParents(2);
        //this.probNet.addConstraint(new NoCycle(), true);
        this.probNet.addConstraint(maxNumParentsConstraint);
        this.probNet.removeConstraint(new DistinctLinks());
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
        double bestPartialScore = currentAccuracy;
        BaseLinkEdit bestEdit = null;
        LearningEditProposal bestEditProposal = null;
        Node bestParent = null;

        for (Node nodeParent : subtractListFromNonRootNode(superParents)) {
            AddLinkEdit addLink = new AddLinkEdit(probNet, getRootNode().getVariable(), nodeParent.getVariable(), true);
            double addScore = metric.getScore(addLink);

            if (!isEditAlreadyConsidered(addLink) && !isBlocked(addLink)
                    && (!onlyAllowedEdits || addLink.getNodeFrom() == getRootNode() || isAllowed(addLink))
                    && (addScore >= bestPartialScore || !onlyPositiveEdits)
            ) {
                bestPartialScore = addScore;
                bestParent = nodeParent;
            }
        }

        if (bestParent != null && !orphans.isEmpty()) {
            superParents.add(bestParent);
            final Node selectedParent = bestParent;
            if (this.sameSP) {
                // sameSP mode: connect bestParent to ALL orphans via a compound edit
                List<PNEdit> linkEdits = orphans.stream()
                        .map(orphan -> (PNEdit) new AddLinkEdit(probNet, selectedParent.getVariable(),
                                orphan.getVariable(), true))
                        .toList();
                ListPNEdit compoundEdit = new ListPNEdit(probNet, linkEdits);
                bestEditProposal = new LearningEditProposal(compoundEdit,
                        new ScoreEditMotivation(bestPartialScore));
                orphans.clear();
                currentAccuracy = bestPartialScore;
            } else {
                for (Node nodeChild : orphans) {
                    AddLinkEdit addLink = new AddLinkEdit(probNet, bestParent.getVariable(), nodeChild.getVariable(), true);
                    double addScore = metric.getScore(addLink);

                    if (!isEditAlreadyConsidered(addLink) && !isBlocked(addLink)
                            && (!onlyAllowedEdits || addLink.getNodeFrom() == getRootNode() || isAllowed(addLink))
                            && (addScore > bestPartialScore || !onlyPositiveEdits)
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
            orphans.remove(probNet.getNode(bestEdit.getVariableTo()));
            currentAccuracy = bestPartialScore;
        }

        return bestEditProposal;
    }
    
    
    private List<Node> subtractListFromNonRootNode(Collection<Node> listToSubtract) {
        return getNonRootNodes().stream().filter(n -> !listToSubtract.contains(n)).collect(Collectors.toList());
    }
}
