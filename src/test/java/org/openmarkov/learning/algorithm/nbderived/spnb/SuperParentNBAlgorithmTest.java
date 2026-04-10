package org.openmarkov.learning.algorithm.nbderived.spnb;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.constraint.DistinctLinks;
import org.openmarkov.core.model.network.constraint.MaxNumParents;
import org.openmarkov.core.model.network.potential.TablePotential;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.metric.cmi.accuracy.Accuracy;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link SuperParentNBAlgorithm}.
 *
 * <p>SPNB starts with NB structure, selects super-parent nodes, and
 * adds arcs from super-parents to orphan features. It removes the
 * DistinctLinks constraint to allow duplicate links.
 *
 * @author Manuel Arias
 */
public class SuperParentNBAlgorithmTest {

    private ProbNet probNet;
    private Variable classVar, f1, f2, f3;
    private StubAccuracy metric;

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

        metric = new StubAccuracy();
    }

    private SuperParentNBAlgorithm createAlgorithm(boolean sameSP) {
        SuperParentNBAlgorithm spnb = new SuperParentNBAlgorithm(
                probNet, null, metric, 1.0, sameSP);
        spnb.setClassVariableName("Class");
        return spnb;
    }

    // --- init ---

    @Test
    public void testInitCreatesNaiveBayesStructure() {
        SuperParentNBAlgorithm spnb = createAlgorithm(false);
        spnb.init(null);

        Node root = probNet.getNode(classVar);
        assertEquals(3, root.getChildren().size(),
                "SPNB should create NB star structure in init");
    }

    @Test
    public void testInitAddsMaxNumParents2() {
        SuperParentNBAlgorithm spnb = createAlgorithm(false);
        spnb.init(null);

        MaxNumParents constraint = probNet.getConstraintOfClass(MaxNumParents.class);
        assertNotNull(constraint);
        assertEquals(2, constraint.getMaxNumParents());
    }

    @Test
    public void testInitRemovesDistinctLinksConstraint() {
        // ProbNet has DistinctLinks by default
        assertTrue(probNet.hasConstraintOfClass(DistinctLinks.class),
                "ProbNet should have DistinctLinks by default");

        SuperParentNBAlgorithm spnb = createAlgorithm(false);
        spnb.init(null);

        assertFalse(probNet.hasConstraintOfClass(DistinctLinks.class),
                "SPNB should remove DistinctLinks constraint");
    }

    // --- getOptimalEdit: super-parent selection ---

    @Test
    public void testGetOptimalEditSelectsSuperParent() {
        // Scores for selecting super-parent (Class->feature edits used for accuracy).
        // With onlyPositiveEdits=true, the score comparison (>= bestPartialScore)
        // drives selection; F2 is the best super-parent at 0.95.
        metric.setScore("Class", "F1", 0.80);
        metric.setScore("Class", "F2", 0.95);
        metric.setScore("Class", "F3", 0.70);
        // Orphan-arc scores must exceed the super-parent score (0.95)
        // for an edit to be proposed.
        metric.setScore("F2", "F1", 0.98);
        metric.setScore("F2", "F3", 0.88);

        SuperParentNBAlgorithm spnb = createAlgorithm(false);
        spnb.init(null);

        LearningEditProposal proposal = spnb.getBestEdit(false, true);
        assertNotNull(proposal);

        // SPNB first selects the best super-parent (F2), then proposes
        // an arc from super-parent to the best orphan (F2->F1, score 0.98).
        AddLinkEdit edit = (AddLinkEdit) proposal.getEdit();
        assertEquals("F2", edit.getVariableFrom().getName(),
                "Best super-parent should be F2 (highest accuracy = 0.95)");
    }

    // --- getOptimalEdit: no proposal when scores don't improve ---

    @Test
    public void testReturnsNullWhenNoImprovementPossible() {
        // All scores are 0 (default), lower than currentAccuracy (0)
        // The condition addScore >= bestPartialScore starts at currentAccuracy=0
        // and addScore > bestPartialScore for orphan arcs, so 0 > 0 is false
        SuperParentNBAlgorithm spnb = createAlgorithm(false);
        spnb.init(null);

        LearningEditProposal proposal = spnb.getBestEdit(true, true);
        assertNull(proposal,
                "Should return null when no edit improves accuracy");
    }

    // --- Stub ---

    private static class StubAccuracy extends Accuracy {
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

        @Override
        protected void initCache() {
            // No-op
        }
    }
}
