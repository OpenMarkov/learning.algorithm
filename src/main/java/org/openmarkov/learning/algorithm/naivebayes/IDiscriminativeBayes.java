package org.openmarkov.learning.algorithm.naivebayes;

import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.Variable;

import java.util.List;

/**
 * Interface for discriminative Bayesian classifiers that operate around a designated root (class) variable.
 */
public interface IDiscriminativeBayes {

    /**
     * Sets the standard NB net given a root node
     */
    void setRelationsForRootVariable();

    /**
     * Returns those nodes which are not been selected by the user as the root of NB
     * @return List of nodes
     */
    List<Node> getNonRootNodes();

    /**
     * Returns the root node that has been selected for this net
     * @return root node
     */
    Node getRootNode();

    /**
     * Returns the variable associated to the root node.
     *
     * @return the root variable
     */
    default Variable getRootVariable() {
        return getRootNode().getVariable();
    }

    /**
     * Returns the list of variables that have not been selected as root of NB
     * @return List of variables
     */
    default List<Variable> getNonRootVariables() {
        return getNonRootNodes().stream().map(Node::getVariable).toList();
    }

}
