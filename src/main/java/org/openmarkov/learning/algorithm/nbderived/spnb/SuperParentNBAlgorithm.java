package org.openmarkov.learning.algorithm.nbderived.spnb;

import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.constraint.DistinctLinks;
import org.openmarkov.core.model.network.constraint.MaxNumParents;
import org.openmarkov.learning.metric.Metric;
import org.openmarkov.learning.core.algorithm.LearningAlgorithmType;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ModelNetUse;
import org.openmarkov.learning.metric.cmi.accuracy.Accuracy;
import org.openmarkov.learning.algorithm.nbderived.common.DiscriminativeAlgorithm;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;


@LearningAlgorithmType(name = "Superparent naive bayes", discriminative = true, supportsUnobservedVariables = false)
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
        final double[] bestPartialScore = {currentAccuracy};
        final BaseLinkEdit[] bestEdit = {null};
        LearningEditProposal bestEditProposal = null;
        Node[] bestParent = {null};
        
        subtractListFromNonRootNode(superParents).forEach(nodeParent -> {
            AddLinkEdit addLink = new AddLinkEdit(probNet, getRootNode().getVariable(), nodeParent.getVariable(), true);
            double addScore = metric.getScore(addLink);
            
            if (!isEditAlreadyConsidered(addLink) && !isBlocked(addLink)
                    && (!onlyAllowedEdits || addLink.getNodeFrom() == getRootNode() || isAllowed(addLink))
                    && (addScore >= bestPartialScore[0] || !onlyPositiveEdits)
            ) {
                bestPartialScore[0] = addScore;
                bestParent[0] = (nodeParent);
            }
        });
        
        if (bestParent[0] != null && !orphans.isEmpty() && !this.sameSP) {
            superParents.add(bestParent[0]);
            orphans.forEach(nodeChild -> {
                AddLinkEdit addLink = new AddLinkEdit(probNet, bestParent[0].getVariable(), nodeChild.getVariable(), true);
                double addScore = metric.getScore(addLink);
                
                if (!isEditAlreadyConsidered(addLink) && !isBlocked(addLink)
                        && (!onlyAllowedEdits || addLink.getNodeFrom() == getRootNode() || isAllowed(addLink))
                        && (addScore > bestPartialScore[0] || !onlyPositiveEdits)
                ) {
                    bestEdit[0] = addLink;
                    bestPartialScore[0] = addScore;
                }
            });
        }
        if (this.sameSP) {
            getNonRootNodes().stream()
                             .filter(n -> !n.getName().equals(bestEdit[0].getVariableTo().getName()))
                             .map(Node::getVariable)
                             .forEach(v -> {
                                 probNet.addLink(bestEdit[0].getVariableTo(), v, true);
                             });
            bestEdit[0] = null;
        }
        
        if (bestEdit[0] != null) {
            bestEditProposal = new SuperParentNaiveBayesEditProposal(bestEdit[0], bestPartialScore[0]);
            markEditAsConsidered(bestEdit[0]);
            orphans.remove(probNet.getNode(bestEdit[0].getVariableTo()));
            currentAccuracy = bestPartialScore[0];
        }
        
        return bestEditProposal;
    }
    
    
    private List<Node> subtractListFromNonRootNode(Collection<Node> listToSubtract) {
        return getNonRootNodes().stream().filter(n -> !listToSubtract.contains(n)).collect(Collectors.toList());
    }
}
