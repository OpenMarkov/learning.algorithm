package org.openmarkov.learning.algorithm.nbderived.common;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.graph.Graph;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.learning.algorithm.naivebayes.IDiscriminativeBayes;
import org.openmarkov.learning.algorithm.scoreAndSearch.ScoreAndSearchAlgorithm;
import org.openmarkov.learning.core.util.LearningEditMotivation;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;
import org.openmarkov.learning.metric.Metric;
import java.util.*;
import java.util.stream.Collectors;


/**
 * Abstract base class for discriminative Bayesian classifier algorithms that extend Naive Bayes
 * with additional inter-feature links. Uses a maximum weight spanning tree (Chow-Liu) for
 * determining feature dependencies.
 */
public abstract class DiscriminativeAlgorithm extends ScoreAndSearchAlgorithm implements IDiscriminativeBayes {

    /**
     * Maximum Weight Spanning Tree for Chow-Liu's algorithm
     */
    protected List<BaseLinkEdit> maximumWeightSpanningTree = new ArrayList<>();

    protected List<BaseLinkEdit> directedMaxWeightSpanningTree = new ArrayList<>();

    /**
     * Manages the history of edits already considered in the current search cycle,
     * preventing the algorithm from re-proposing the same edit.
     */
    protected final EditHistorySupport editHistory = new EditHistorySupport(new ArrayList<>());

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



    /**
     * Computes and initializes the maximum weight spanning tree based on the Chow-Liu algorithm
     */
    protected void buildMaximumWeightSpanningTree(){
        Map<Set<Node>, Double> map = new HashMap<Set<Node>, Double>();
        Graph auxTree = new Graph<>();

        this.getNonRootNodes().forEach(n1 ->{
            this.getNonRootNodes().forEach(n-> auxTree.addNode(n1));
            this.getNonRootNodes().stream().filter(n -> n!=n1).forEach(n2->{
                Set<Node> nodes = new HashSet<>(Arrays.asList(n1, n2));
                if(!map.containsKey(nodes)){
                    map.put(nodes, metric.getScore(new AddLinkEdit(probNet, n1.getVariable(), n2.getVariable(), false)));
                }
            });
        });

      map.entrySet().stream().sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                             .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue,
                                     (oldValue, newValue) -> oldValue, LinkedHashMap::new))
              .keySet().forEach(k-> {
                            List<Node> nodes = new ArrayList(k);
                            if(!auxTree.existsPath(nodes.get(0), nodes.get(1), false, Collections.emptyList())){
                                auxTree.addLink(nodes.get(0), nodes.get(1), false);
                                maximumWeightSpanningTree.add(new AddLinkEdit(probNet, nodes.get(0).getVariable(), nodes.get(1).getVariable(), false));
                            }
        });

    }

    /**
     * Checks whether given link belongs to the maximum weight spanning tree structure
     * @param v1 First variable
     * @param v2 Second variable
     * @return true if the link belongs to the maximum weight spanning tree
     */
    protected boolean withinMaxWeightSpanningTree(Variable v1, Variable v2){
        return directedMaxWeightSpanningTree.stream()
                                            .anyMatch(l -> (l.getVariableFrom() == v1 && l.getVariableTo() == v2));
    }


    /**
     * Redirects the maximum weight spanning tree edges to point away from the given root variable.
     *
     * @param root the variable to use as the root of the directed tree
     * @return a list of directed link edits forming the redirected tree
     */
    protected List<BaseLinkEdit> redirectMaximumWeightSpanningTree(Variable root){
        List<BaseLinkEdit> redirectedTree = new ArrayList<>();
        List<Variable> nodes = new ArrayList<>();

        nodes.add(root);

        while(!nodes.isEmpty()){
            Variable head = nodes.get(0);
            getEditsForVariable(maximumWeightSpanningTree, head).forEach(edit -> {
                BaseLinkEdit directedEdit = new AddLinkEdit(probNet, head, edit.getVariableFrom() == head ? edit.getVariableTo() : edit.getVariableFrom(), true);
                if (redirectedTree.stream()
                                  .noneMatch(t -> t.getVariableTo()
                                                   .equals(directedEdit.getVariableFrom()) && t.getVariableFrom()
                                                                                               .equals(directedEdit.getVariableTo())
                                          || t.getVariableFrom()
                                              .equals(directedEdit.getVariableFrom()) && t.getVariableTo()
                                                                                          .equals(directedEdit.getVariableTo())
                )){
                    nodes.add(directedEdit.getVariableTo());
                    redirectedTree.add(directedEdit);
                }
            });
            nodes.remove(head);
        }
        return redirectedTree;
    }
    
    private static List<BaseLinkEdit> getEditsForVariable(List<BaseLinkEdit> list, Variable v) {
        return list.stream()
                   .filter(edit -> edit.getVariableFrom() == v || edit.getVariableTo() == v)
                   .collect(Collectors.toList());
    }


    /**
     * Returns a randomly selected non-root variable.
     *
     * @return a random non-root variable
     */
    protected Variable getRandomVariable(){
        return getNonRootVariables().get(new Random().nextInt(getNonRootVariables().size()-1));
    }

    @Override
    public Node getRootNode(){
        return this.probNet.getNodes().stream().filter(n-> n.getVariable().getName().equals(classVariableName)).findFirst().get();
    }

    @Override
    public List<Node> getNonRootNodes(){
        return  this.probNet.getNodes().stream().filter(n-> !n.getVariable().getName().equals(classVariableName)).collect(Collectors.toList());
    }

    @Override
    public List<Variable> getNonRootVariables(){
        return this.getNonRootNodes().stream().map(Node::getVariable).collect(Collectors.toList());
    }


    @Override
    public Variable getRootVariable(){
        return getRootNode().getVariable();
    }



}
