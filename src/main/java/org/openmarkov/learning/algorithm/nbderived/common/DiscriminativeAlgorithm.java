package org.openmarkov.learning.algorithm.nbderived.common;

import org.openmarkov.core.action.AddLinkEdit;
import org.openmarkov.core.action.BaseLinkEdit;
import org.openmarkov.core.action.PNEdit;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.graph.Graph;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.learning.algorithm.naivebayes.IDiscriminativeBayes;
import org.openmarkov.learning.algorithm.scoreAndSearch.ScoreAndSearchAlgorithm;
import org.openmarkov.learning.metric.Metric;
import java.util.*;
import java.util.stream.Collectors;


public abstract class DiscriminativeAlgorithm extends ScoreAndSearchAlgorithm implements IDiscriminativeBayes {

    /**
     * Maximum Weight Spanning Tree for Chow-Liu's algorithm
     */
    protected List<BaseLinkEdit> maximumWeightSpanningTree = new ArrayList<>();

    protected List<BaseLinkEdit> directedMaxWeightSpanningTree = new ArrayList<>();

    /**
     * List with the best edits that have been not been done by the
     * algorithm because they violate the ModelNetworkConstraint
     */
    protected List<PNEdit> lastBestEdits;

    /**
     * Metric used to compute the Conditional Mutual Information for each pair of nodes conditioned to the class variable
     */
    protected Metric unconditionedMetric;



    public DiscriminativeAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Metric metric, Double alpha) {
        super(probNet, caseDatabase, metric, alpha);
        this.lastBestEdits = new ArrayList<PNEdit>();
    }


    /**
     * Sets the standard NB net given a root node
     */
    public void setRelationsForRootVariable() {
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
                            if(!auxTree.existsPath(nodes.get(0), nodes.get(1), false)){
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
        return directedMaxWeightSpanningTree.stream().anyMatch(l -> (l.getVariable1()==v1 && l.getVariable2()==v2));
    }


    protected List<BaseLinkEdit> redirectMaximumWeightSpanningTree(Variable root){
        List<BaseLinkEdit> redirectedTree = new ArrayList<>();
        List<Variable> nodes = new ArrayList<>();

        nodes.add(root);

        while(!nodes.isEmpty()){
            Variable head = nodes.get(0);
            getEditsForVariable(maximumWeightSpanningTree, head).forEach(edit -> {
                BaseLinkEdit directedEdit = new AddLinkEdit(probNet, head, edit.getVariable1()==head?edit.getVariable2():edit.getVariable1(), true);
                if(redirectedTree.stream().noneMatch(t -> t.getVariable2().equals(directedEdit.getVariable1()) && t.getVariable1().equals(directedEdit.getVariable2())
                        || t.getVariable1().equals(directedEdit.getVariable1()) && t.getVariable2().equals(directedEdit.getVariable2())
                )){
                    nodes.add(directedEdit.getVariable2());
                    redirectedTree.add(directedEdit);
                }
            });
            nodes.remove(head);
        }
        return redirectedTree;
    }

    private List<BaseLinkEdit> getEditsForVariable(List<BaseLinkEdit> list, Variable v){
        return  list.stream().filter(edit -> edit.getVariable1()==v || edit.getVariable2()==v).collect(Collectors.toList());
    }


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
