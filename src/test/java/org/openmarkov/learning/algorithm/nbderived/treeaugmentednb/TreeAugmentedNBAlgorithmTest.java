package org.openmarkov.learning.algorithm.nbderived.treeaugmentednb;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.constraint.MaxNumParents;
import org.openmarkov.core.model.network.constraint.NoCycle;
import org.openmarkov.core.model.network.potential.TablePotential;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.metric.Metric;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link TreeAugmentedNBAlgorithm}.
 *
 * <p>Uses a testable subclass that fixes the random variable for MWST
 * redirect, making the directed tree deterministic.
 *
 * @author Manuel Arias
 */
public class TreeAugmentedNBAlgorithmTest {

    private ProbNet probNet;
    private Variable classVar, f1, f2, f3;
    private StubMetric metric;

    @BeforeEach
    public void setup() {
        probNet = new ProbNet();
        classVar = new Variable("Class", 2);
        f1 = new Variable("F1", 2);
        f2 = new Variable("F2", 2);
        f3 = new Variable("F3", 2);
        probNet.addNode(classVar, NodeType.CHANCE);
        probNet.addNode(f1, NodeType.CHANCE);
        probNet.addNode(f2, NodeType.CHANCE);
        probNet.addNode(f3, NodeType.CHANCE);

        metric = new StubMetric();
        // MWST scores (undirected feature pairs): F1-F2 > F1-F3 > F2-F3
        // MWST should be: F1-F2, F1-F3
        metric.setScore("F1", "F2", 10.0);
        metric.setScore("F1", "F3", 5.0);
        metric.setScore("F2", "F3", 3.0);
    }

    private TestableTAN createAlgorithm() {
        // Fix the random variable to F1 so the directed MWST is: F1->F2, F1->F3
        TestableTAN tan = new TestableTAN(probNet, null, metric, 1.0, f1);
        tan.setClassVariableName("Class");
        return tan;
    }

    // --- init ---

    @Test
    public void testInitCreatesStarStructureAndConstraints() {
        TestableTAN tan = createAlgorithm();
        tan.init(null);

        Node root = probNet.getNode(classVar);
        assertEquals(3, root.getChildren().size(), "Class should have 3 children");
        assertTrue(probNet.hasConstraintOfClass(NoCycle.class));
        assertTrue(probNet.hasConstraintOfClass(MaxNumParents.class));

        MaxNumParents constraint = probNet.getConstraintOfClass(MaxNumParents.class);
        assertEquals(2, constraint.getMaxNumParents(), "TAN uses MaxNumParents(2)");
    }

    @Test
    public void testInitBuildsMWST() {
        TestableTAN tan = createAlgorithm();
        tan.init(null);

        // After init, directedMaxWeightSpanningTree should have N-1 = 2 edges
        assertEquals(2, tan.getDirectedMWSTSize(),
                "Directed MWST should have exactly 2 edges for 3 features");
    }

    // --- getOptimalEdit with MWST filtering ---

    @Test
    public void testGetOptimalEditOnlyProposesWithinMWST() {
        TestableTAN tan = createAlgorithm();
        tan.init(null);

        // Collect all proposed edits (with onlyAllowedEdits=true for MWST filtering)
        Set<String> proposedEdges = new HashSet<>();
        LearningEditProposal proposal = tan.getBestEdit(true, false);
        while (proposal != null) {
            AddLinkEdit edit = (AddLinkEdit) proposal.getEdit();
            proposedEdges.add(edit.getVariableFrom().getName() + "->" + edit.getVariableTo().getName());
            proposal = tan.getNextEdit(true, false);
        }

        // With root=F1, directed MWST is F1->F2 and F1->F3
        assertEquals(Set.of("F1->F2", "F1->F3"), proposedEdges,
                "All proposed edits should be edges in the directed MWST");
    }

    @Test
    public void testGetOptimalEditReturnsNullWhenAllConsidered() {
        TestableTAN tan = createAlgorithm();
        tan.init(null);

        // Exhaust all MWST edges
        tan.getBestEdit(true, false);
        tan.getNextEdit(true, false);

        LearningEditProposal proposal = tan.getNextEdit(true, false);
        assertNull(proposal, "Should return null when all MWST edges have been proposed");
    }

    @Test
    public void testGetOptimalEditWithoutConstraintCheckProposesAnyFeatureEdge() {
        TestableTAN tan = createAlgorithm();
        tan.init(null);

        // With onlyAllowedEdits=false, MWST filtering is skipped
        LearningEditProposal proposal = tan.getBestEdit(false, false);
        assertNotNull(proposal);

        // Should be a feature->feature edit (not class-related)
        AddLinkEdit edit = (AddLinkEdit) proposal.getEdit();
        assertNotEquals("Class", edit.getVariableFrom().getName());
        assertNotEquals("Class", edit.getVariableTo().getName());
    }

    // --- Helpers ---

    private static class TestableTAN extends TreeAugmentedNBAlgorithm {
        private final Variable fixedRoot;

        TestableTAN(ProbNet probNet, Object caseDatabase, Metric metric, Double alpha, Variable fixedRoot) {
            super(probNet, null, metric, alpha);
            this.fixedRoot = fixedRoot;
        }

        @Override
        protected Variable getRandomVariable() {
            return fixedRoot;
        }

        int getDirectedMWSTSize() {
            return mwst.getDirectedEdges().size();
        }
    }

    private static class StubMetric extends Metric {
        private final Map<String, Double> scores = new HashMap<>();

        void setScore(String from, String to, double score) {
            scores.put(from + ":" + to, score);
        }

        @Override
        public double getScore(PNEdit edit) {
            if (edit instanceof BaseLinkEdit linkEdit) {
                String key = linkEdit.getVariableFrom().getName() + ":" + linkEdit.getVariableTo().getName();
                return scores.getOrDefault(key, 0.0);
            }
            return 0.0;
        }

        @Override
        public double score(TablePotential nodePotential) {
            return 0;
        }
    }
}
