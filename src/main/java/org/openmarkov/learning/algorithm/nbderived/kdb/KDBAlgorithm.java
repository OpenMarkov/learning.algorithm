package org.openmarkov.learning.algorithm.nbderived.kdb;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
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
import org.openmarkov.learning.algorithm.nbderived.kdb.util.KDBEditProposal;
import org.openmarkov.learning.metric.cmi.mutualInformation.MutualInformationMetric;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;


@LearningAlgorithmType(name = "K-Dependence bayesian classifier", discriminative = true, supportsUnobservedVariables = false)
public class KDBAlgorithm extends DiscriminativeAlgorithm {
    
    
    /**
     * KDB Algorithm
     * 1. For each feature Xi, compute mutual information I(Xi;C) where C is the class
     * 2. Compute class conditional mutual information I(Xi; Xj|C) for each pair of features Xi and Xj where i!=j
     * 3. Let the used variable list, S be empty
     * 4. Let the Bayesian network being constructed, BN, begin with a single class node, C
     * 5. Repeat until S includes all domain features
     * 	5.1. Select feature Xmax which is not in S and has the largest value I(Xmax;C)
     * 	5.2. Add a node to BN representing Xmax
     * 	5.3. Add an arc from C to Xmax in BN
     * 	5.4. Add m=min(|S|,k) arcs from m distinct features Xj in S with the highest value for I(max;Xj|C).
     * 	5.5. Add Xmax to S
     */
    
    
    /**
     * List with the best edits that have been not been done by the
     * algorithm because they violate the ModelNetworkConstraint
     */
    protected List<PNEdit> lastBestEdits;
    /**
     * Maximum allowable degree of feature dependence
     */
    private int kDependence;
    /**
     * Domain features - Step 5
     */
    private LinkedList<String> domainFeatures;
    
    
    public KDBAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Metric metric, Metric unconditioned, Double alpha) {
        super(probNet, caseDatabase, metric, alpha);
        this.lastBestEdits = new ArrayList<PNEdit>();
        unconditionedMetric = unconditioned;
        unconditionedMetric.init(probNet, caseDatabase);
    }
    
    public KDBAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Metric metric, Metric unconditioned, Double alpha, int kDependence) {
        this(probNet, caseDatabase, metric, unconditioned, alpha);
        this.kDependence = Math.min(kDependence, probNet.getNumNodes() - 2);
    }
    
    
    /**
     * This method returns the best edit (and its associated score)
     * that can be done to the network that is being learnt.
     *
     * @param onlyAllowedEdits  If this parameter is true, only those edits
     *                          that do not provoke a ConstraintViolatedException are returned
     * @param onlyPositiveEdits If this parameter is true, only those
     *                          edits with a positive associated score are returned.
     * @return {@code LearningEditProposal} with the best edit and its score.
     */
    @Override public LearningEditProposal getBestEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        resetHistory();
        return getNextEdit(onlyAllowedEdits, onlyPositiveEdits);
    }
    
    
    protected void resetHistory() {
        lastBestEdits.clear();
    }
    
    
    @Override public LearningEditMotivation getMotivation(PNEdit edit) {
        return new ScoreEditMotivation(
                (((BaseLinkEdit) edit).getVariable1()
                                      .getName() == getRootNode().getName() ? unconditionedMetric : metric).getScore(edit)
        );
    }
    
    
    /**
     * This method returns the next best edit (and its associated score)
     * that can be done to the network that is being learnt.
     *
     * @param onlyAllowedEdits  If this parameter is true, only those edits
     *                          that do not provoke a ConstraintViolatedException are returned
     * @param onlyPositiveEdits If this parameter is true, only those
     *                          edits with a positive associated score are returned.
     * @return {@code LearningEditProposal} with the best edit and its score.
     */
    @Override public LearningEditProposal getNextEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        return getOptimalEdit(probNet, onlyAllowedEdits, onlyPositiveEdits);
    }
    
    @Override public void init(ModelNetUse modelNetUse) {
        domainFeatures = new LinkedList<>();
        setRelationsForRootVariable();
        if (metric instanceof MutualInformationMetric miMetric) {
            miMetric.setClassVariable(this.classVariableName);
        }
        if (unconditionedMetric instanceof MutualInformationMetric miMetric) {
            miMetric.setClassVariable(this.classVariableName);
        }
        MaxNumParents maxNumParentsConstraint = new MaxNumParents();
        maxNumParentsConstraint.setMaxNumParents(kDependence + 1);
        this.probNet.addConstraint(new NoCycle());
        this.probNet.addConstraint(maxNumParentsConstraint);
    }
    
    /**
     * Store last best edit
     *
     * @param edit
     */
    protected void markEditAsConsidered(BaseLinkEdit edit) {
        lastBestEdits.add(edit);
    }
    
    /**
     * @param edit
     * @return true if it is an edit already considered
     */
    protected boolean isEditAlreadyConsidered(BaseLinkEdit edit) {
        return lastBestEdits.contains(edit);
    }
    
    
    @Override
    protected void buildMaximumWeightSpanningTree() {
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
        
        Node xMaxWithPendingArcs = getLastXMaxWithPendingArcs();
        
        if (needToComputeXMax(xMaxWithPendingArcs)) {
            bestEdit[0] = getNewXMaxFeatureEdit();
            if (bestEdit[0] != null) {
                bestPartialScore[0] = unconditionedMetric.getScore(bestEdit[0]);
                domainFeatures.add(bestEdit[0].getVariable2().getName());
            }
            
        } else {
            domainFeatures.stream().filter(n -> n != xMaxWithPendingArcs.getVariable().getName()).forEach(df -> {
                AddLinkEdit addLink = new AddLinkEdit(probNet, probNet.getVariable(df), xMaxWithPendingArcs.getVariable(), true);
                double addScore = metric.getScore(addLink);
                
                if ((addScore >= bestPartialScore[0]) && !isEditAlreadyConsidered(addLink)
                        && ((!onlyAllowedEdits || isAllowed(addLink))
                        && (!onlyPositiveEdits || addScore >= 0) && !isBlocked(addLink))
                ) {
                    bestEdit[0] = addLink;
                    bestPartialScore[0] = addScore;
                }
                
            });
        }
        
        if (bestEdit[0] != null) {
            bestEditProposal = new KDBEditProposal(bestEdit[0], bestPartialScore[0]);
            markEditAsConsidered(bestEdit[0]);
        }
        return bestEditProposal;
    }
    
    /**
     * Checks whether we need to compute a new Xmax -- step 5.1. KDB algorithm
     *
     * @param xMax
     * @return
     */
    private boolean needToComputeXMax(Node xMax) {
        return xMax == null || (xMax.getNumParents() - 1) >= Math.min(kDependence, domainFeatures.size() - 1);
    }
    
    
    /**
     * Returns the last node that has been added only if there are parent-connections pending for this node
     *
     * @return Node to connect
     */
    private Node getLastXMaxWithPendingArcs() {
        Node node = !domainFeatures.isEmpty() && (probNet.getNode(domainFeatures.getLast())
                                                         .getNumParents() - 1) < Math.min(kDependence, domainFeatures.size() - 1) ?
                probNet.getNode(domainFeatures.getLast()) : null;
        return node;
    }
    
    
    /**
     * Provides the feature not in the domainFeature list that returns the highest IM value
     *
     * @return Best Class-XMaxFeature edit
     */
    private BaseLinkEdit getNewXMaxFeatureEdit() {
        
        final Node[] xMax = {null};
        final double[] maxScore = {0.0};
        
        if (getNonRootNodes().size() > domainFeatures.size()) {
            getNonRootNodes().forEach(node -> {
                if (!domainFeatures.contains(node.getVariable().getName())) {
                    double score = unconditionedMetric.getScore(new AddLinkEdit(probNet, this.getRootNode()
                                                                                             .getVariable(), node.getVariable(), true));
                    
                    if (score > maxScore[0]) {
                        maxScore[0] = score;
                        xMax[0] = node;
                    }
                }
            });
        }
        
        return xMax[0] != null ?
                new AddLinkEdit(probNet, this.getRootNode().getVariable(), xMax[0].getVariable(), true)
                : null;
    }
    
    
}
