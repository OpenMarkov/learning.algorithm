package org.openmarkov.learning.algorithm.naivebayes;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.learning.core.util.ModelNetUse;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link NaiveBayesAlgorithm}.
 *
 * @author Manuel Arias
 */
public class NaiveBayesAlgorithmTest {

    private ProbNet probNet;
    private NaiveBayesAlgorithm nb;

    @BeforeEach
    public void setup() {
        probNet = new ProbNet();
        probNet.addNode(new Variable("Class"), NodeType.CHANCE);
        probNet.addNode(new Variable("F1"), NodeType.CHANCE);
        probNet.addNode(new Variable("F2"), NodeType.CHANCE);
        probNet.addNode(new Variable("F3"), NodeType.CHANCE);

        nb = new NaiveBayesAlgorithm(probNet, null, 1.0);
        nb.setClassVariableName("Class");
    }

    @Test
    public void testGetRootNode() {
        assertEquals("Class", nb.getRootNode().getVariable().getName());
    }

    @Test
    public void testGetNonRootNodes() {
        Set<String> names = nb.getNonRootNodes().stream()
                               .map(n -> n.getVariable().getName())
                               .collect(Collectors.toSet());
        assertEquals(Set.of("F1", "F2", "F3"), names);
    }

    @Test
    public void testInitCreatesStarStructure() {
        nb.init(null);

        Node root = nb.getRootNode();
        // Root should have 3 children (directed links to all features)
        assertEquals(3, root.getChildren().size());

        // Each feature should have exactly 1 parent: the class
        for (Node feature : nb.getNonRootNodes()) {
            List<Node> parents = feature.getParents();
            assertEquals(1, parents.size());
            assertEquals("Class", parents.get(0).getVariable().getName());
        }
    }

    @Test
    public void testNoFeatureInterLinks() {
        nb.init(null);

        // No links between features
        List<Node> features = nb.getNonRootNodes();
        for (Node f1 : features) {
            for (Node f2 : features) {
                if (f1 != f2) {
                    assertNull(probNet.getLink(f1, f2, true),
                            f1.getName() + " should not link to " + f2.getName());
                }
            }
        }
    }

    @Test
    public void testStructuralMethodsReturnNull() {
        assertNull(nb.getBestEdit(false, false));
        assertNull(nb.getNextEdit(false, false));
        assertNull(nb.getMotivation(null));
    }

    @Test
    public void testDefaultInterfaceMethods() {
        // getRootVariable and getNonRootVariables come from IDiscriminativeBayes defaults
        assertEquals("Class", nb.getRootVariable().getName());

        Set<String> varNames = nb.getNonRootVariables().stream()
                                  .map(Variable::getName)
                                  .collect(Collectors.toSet());
        assertEquals(Set.of("F1", "F2", "F3"), varNames);
    }
}
