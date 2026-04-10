package org.openmarkov.learning.algorithm.nbderived.snb;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.action.base.linkEdits.RemoveLinkEdit;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.constraint.MaxNumParents;
import org.openmarkov.core.model.network.constraint.NoCycle;
import org.openmarkov.core.model.network.potential.TablePotential;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.metric.cmi.accuracy.Accuracy;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link SelectiveNBAlgorithm}.
 *
 * <p>SNB operates in two modes:
 * <ul>
 *   <li><b>Forward:</b> starts with no feature links, proposes AddLinkEdit</li>
 *   <li><b>Backward:</b> starts with NB structure, proposes RemoveLinkEdit</li>
 * </ul>
 *
 * @author Manuel Arias
 */
public class SelectiveNBAlgorithmTest {

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

    private SelectiveNBAlgorithm createForward() {
        SelectiveNBAlgorithm snb = new SelectiveNBAlgorithm(probNet, null, metric, 1.0, true);
        snb.setClassVariableName("Class");
        return snb;
    }

    private SelectiveNBAlgorithm createBackward() {
        SelectiveNBAlgorithm snb = new SelectiveNBAlgorithm(probNet, null, metric, 1.0, false);
        snb.setClassVariableName("Class");
        return snb;
    }

    // --- init: forward mode ---

    @Test
    public void testInitForward_NoFeatureLinks() {
        SelectiveNBAlgorithm snb = createForward();
        snb.init(null);

        Node root = probNet.getNode(classVar);
        assertEquals(0, root.getChildren().size(),
                "Forward mode should not create feature links in init");
    }

    @Test
    public void testInitForward_Constraints() {
        SelectiveNBAlgorithm snb = createForward();
        snb.init(null);

        assertTrue(probNet.hasConstraintOfClass(NoCycle.class));
        MaxNumParents constraint = probNet.getConstraintOfClass(MaxNumParents.class);
        assertNotNull(constraint);
        assertEquals(1, constraint.getMaxNumParents(),
                "SNB uses MaxNumParents(1)");
    }

    // --- init: backward mode ---

    @Test
    public void testInitBackward_NaiveBayesStructure() {
        SelectiveNBAlgorithm snb = createBackward();
        snb.init(null);

        Node root = probNet.getNode(classVar);
        assertEquals(3, root.getChildren().size(),
                "Backward mode should create NB star structure");
    }

    // --- getOptimalEdit: forward mode ---

    @Test
    public void testForwardMode_ProposesAddLink() {
        metric.setScore("Class", "F1", 0.85);
        metric.setScore("Class", "F2", 0.90);
        metric.setScore("Class", "F3", 0.80);

        SelectiveNBAlgorithm snb = createForward();
        snb.init(null);

        LearningEditProposal proposal = snb.getBestEdit(false, false);
        assertNotNull(proposal);
        assertInstanceOf(AddLinkEdit.class, proposal.getEdit(),
                "Forward mode should propose AddLinkEdit");

        AddLinkEdit edit = (AddLinkEdit) proposal.getEdit();
        assertEquals("Class", edit.getVariableFrom().getName());
    }

    // --- getOptimalEdit: backward mode ---

    @Test
    public void testBackwardMode_ProposesRemoveLink() {
        metric.setScore("Class", "F1", 0.85);
        metric.setScore("Class", "F2", 0.90);
        metric.setScore("Class", "F3", 0.80);

        SelectiveNBAlgorithm snb = createBackward();
        snb.init(null);

        LearningEditProposal proposal = snb.getBestEdit(false, false);
        assertNotNull(proposal);
        assertInstanceOf(RemoveLinkEdit.class, proposal.getEdit(),
                "Backward mode should propose RemoveLinkEdit");

        RemoveLinkEdit edit = (RemoveLinkEdit) proposal.getEdit();
        assertEquals("Class", edit.getVariableFrom().getName());
    }

    // --- forward candidate set ---

    @Test
    public void testForwardCandidatesAreUnconnectedFeatures() {
        metric.setScore("Class", "F2", 0.90);

        SelectiveNBAlgorithm snb = createForward();
        snb.init(null);

        // In forward mode, candidates are features NOT yet connected to class.
        // Initially all 3 features are unconnected.
        LearningEditProposal proposal = snb.getBestEdit(false, false);
        assertNotNull(proposal, "Should propose an edit for unconnected features");
    }

    // --- backward candidate set ---

    @Test
    public void testBackwardCandidatesAreConnectedFeatures() {
        metric.setScore("Class", "F1", 0.85);
        metric.setScore("Class", "F2", 0.90);
        metric.setScore("Class", "F3", 0.80);

        SelectiveNBAlgorithm snb = createBackward();
        snb.init(null);

        // In backward mode, candidates are features connected to class (all 3 after init)
        LearningEditProposal proposal = snb.getBestEdit(false, false);
        assertNotNull(proposal, "Should propose removal of connected features");
    }

    // --- Stub ---

    /**
     * Stub extending Accuracy so that SNB's cast to Accuracy succeeds.
     * Overrides getScore and initCache to avoid needing real CaseDatabase.
     */
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
            // No-op: avoid NPE on null caseDatabase
        }
    }
}
