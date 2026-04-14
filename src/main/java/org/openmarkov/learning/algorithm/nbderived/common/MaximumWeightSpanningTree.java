package org.openmarkov.learning.algorithm.nbderived.common;

import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.model.graph.Graph;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.learning.metric.Metric;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Computes and manages a Maximum Weight Spanning Tree (Chow-Liu) over the
 * feature nodes of a Bayesian classifier.
 * <p>
 * The tree is built using Kruskal's algorithm: all pairs of non-root feature
 * nodes are scored with a metric (typically conditional mutual information),
 * then edges are added in decreasing score order as long as they do not create
 * a cycle. The resulting undirected tree can then be directed away from a
 * chosen root variable.
 *
 * @author Manuel Arias
 */
public class MaximumWeightSpanningTree {

    private final List<BaseLinkEdit> undirectedEdges = new ArrayList<>();
    private List<BaseLinkEdit> directedEdges = new ArrayList<>();

    /**
     * Builds the maximum weight spanning tree over the given feature nodes.
     *
     * @param probNet  the probabilistic network (used to create link edits)
     * @param metric   the scoring metric for evaluating pair-wise feature dependence
     * @param features the non-root (feature) nodes to connect
     */
    public void build(ProbNet probNet, Metric metric, List<Node> features) {
        undirectedEdges.clear();
        directedEdges.clear();

        Map<Set<Node>, Double> scores = new HashMap<>();
        Graph<Node> auxTree = new Graph<>();

        for (Node n1 : features) {
            auxTree.addNode(n1);
            for (Node n2 : features) {
                if (n2 == n1) continue;
                Set<Node> pair = new HashSet<>(Arrays.asList(n1, n2));
                if (!scores.containsKey(pair)) {
                    scores.put(pair, metric.getScore(
                            new AddLinkEdit(probNet, n1.getVariable(), n2.getVariable(), false)));
                }
            }
        }

        // Kruskal: add edges in decreasing score order, skipping those that would form a cycle
        scores.entrySet().stream()
                .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue,
                        (oldValue, newValue) -> oldValue, LinkedHashMap::new))
                .keySet().forEach(pair -> {
                    List<Node> nodes = new ArrayList<>(pair);
                    if (!auxTree.existsPath(nodes.get(0), nodes.get(1), false, Collections.emptyList())) {
                        auxTree.addLink(nodes.get(0), nodes.get(1), false);
                        undirectedEdges.add(new AddLinkEdit(probNet,
                                nodes.get(0).getVariable(), nodes.get(1).getVariable(), false));
                    }
                });
    }

    /**
     * Directs all undirected edges away from the given root variable using BFS.
     *
     * @param root the variable to use as the root of the directed tree
     * @param probNet the network (used to create directed link edits)
     */
    public void redirect(Variable root, ProbNet probNet) {
        directedEdges = new ArrayList<>();
        List<Variable> queue = new ArrayList<>();
        queue.add(root);

        while (!queue.isEmpty()) {
            Variable head = queue.get(0);
            for (BaseLinkEdit edit : edgesIncidentTo(head)) {
                Variable other = edit.getVariableFrom() == head
                        ? edit.getVariableTo() : edit.getVariableFrom();
                BaseLinkEdit directedEdit = new AddLinkEdit(probNet, head, other, true);
                if (directedEdges.stream().noneMatch(t ->
                        (t.getVariableTo().equals(directedEdit.getVariableFrom())
                                && t.getVariableFrom().equals(directedEdit.getVariableTo()))
                                || (t.getVariableFrom().equals(directedEdit.getVariableFrom())
                                && t.getVariableTo().equals(directedEdit.getVariableTo())))) {
                    queue.add(other);
                    directedEdges.add(directedEdit);
                }
            }
            queue.remove(head);
        }
    }

    /**
     * Checks whether the directed tree contains an edge from v1 to v2.
     *
     * @param v1 source variable
     * @param v2 target variable
     * @return true if the directed MWST contains v1 → v2
     */
    public boolean contains(Variable v1, Variable v2) {
        return directedEdges.stream()
                .anyMatch(l -> l.getVariableFrom() == v1 && l.getVariableTo() == v2);
    }

    /**
     * Returns true if the tree has been built (has undirected edges).
     */
    public boolean isBuilt() {
        return !undirectedEdges.isEmpty();
    }

    /**
     * Returns the directed edges of the tree after {@link #redirect} has been called.
     */
    public List<BaseLinkEdit> getDirectedEdges() {
        return Collections.unmodifiableList(directedEdges);
    }

    /**
     * Returns the undirected edges of the tree.
     */
    public List<BaseLinkEdit> getUndirectedEdges() {
        return Collections.unmodifiableList(undirectedEdges);
    }

    private List<BaseLinkEdit> edgesIncidentTo(Variable v) {
        return undirectedEdges.stream()
                .filter(edit -> edit.getVariableFrom() == v || edit.getVariableTo() == v)
                .collect(Collectors.toList());
    }
}
