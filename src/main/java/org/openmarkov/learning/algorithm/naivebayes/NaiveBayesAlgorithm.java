package org.openmarkov.learning.algorithm.naivebayes;

import org.openmarkov.core.action.PNEdit;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.learning.core.algorithm.LearningAlgorithm;
import org.openmarkov.learning.core.algorithm.LearningAlgorithmType;
import org.openmarkov.learning.core.util.LearningEditMotivation;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ModelNetUse;

import java.util.List;
import java.util.stream.Collectors;

@LearningAlgorithmType(name = "Naive bayes", discriminative = true)
public class NaiveBayesAlgorithm extends LearningAlgorithm implements IDiscriminativeBayes {
    
    
    public NaiveBayesAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, double alpha) {
        super(probNet, caseDatabase, alpha);
    }
    
    
    @Override
    public void setRelationsForRootVariable() {
        Node root = this.getRootNode();
        this.getNonRootNodes().stream().forEach(node -> {
            probNet.addLink(root, node, true);
        });
    }
    
    
    @Override public void init(ModelNetUse modelNetUse) {
        setRelationsForRootVariable();
    }
    
    @Override
    public LearningEditProposal getBestEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        return null;
    }
    
    @Override public LearningEditProposal getNextEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        return null;
    }
    
    @Override public LearningEditMotivation getMotivation(PNEdit edit) {
        return null;
    }
    
    @Override
    public Node getRootNode() {
        return this.probNet.getNodes()
                           .stream()
                           .filter(n -> n.getVariable().getName().equals(classVariableName))
                           .findFirst()
                           .get();
    }
    
    @Override
    public List<Node> getNonRootNodes() {
        return this.probNet.getNodes()
                           .stream()
                           .filter(n -> !n.getVariable().getName().equals(classVariableName))
                           .collect(Collectors.toList());
    }
    
    @Override
    public Variable getRootVariable() {
        return null;
    }
    
    @Override
    public List<Variable> getNonRootVariables() {
        return null;
    }
}
