package org.openmarkov.learning.algorithm.hillclimbing;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.action.base.linkEdits.InvertLinkEdit;
import org.openmarkov.core.action.base.linkEdits.RemoveLinkEdit;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.potential.TablePotential;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;
import org.openmarkov.learning.metric.Metric;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class HillClimbingAlgorithmTest {

    private ProbNet probNet;
    private Variable varA;
    private Variable varB;
    private StubMetric metric;
    private HillClimbingAlgorithm hc;

    @BeforeEach
    public void setup() {
        probNet = new ProbNet();
        varA = new Variable("A");
        varB = new Variable("B");
        Variable varC = new Variable("C");
        probNet.addNode(varA, NodeType.CHANCE);
        probNet.addNode(varB, NodeType.CHANCE);
        probNet.addNode(varC, NodeType.CHANCE);

        metric = new StubMetric();
        // CaseDatabase can be null for this test as we stub the metric and don't use it
        // in HC logic
        hc = new HillClimbingAlgorithm(probNet, null, 0.05, metric);
    }

    @Test
    public void testStandardBehavior_AddLink() {
        // Setup: A->B is the best positive edit
        metric.setScore("A", "B", AddLinkEdit.class, 10.0);
        metric.setScore("B", "A", AddLinkEdit.class, 5.0);

        LearningEditProposal proposal = hc.getBestEdit(true, true);
        assertNotNull(proposal, "Should propose an edit");
        assertEquals(10.0, ((ScoreEditMotivation) proposal.getMotivation()).getScore(), 0.001);
        assertInstanceOf(AddLinkEdit.class, proposal.getEdit());
        AddLinkEdit edit = (AddLinkEdit) proposal.getEdit();
        assertEquals("A", edit.getVariableFrom().getName());
        assertEquals("B", edit.getVariableTo().getName());
    }

    @Test
    public void testRespectsOnlyPositive() {
        // Setup: Only negative scores available
        metric.setScore("A", "B", AddLinkEdit.class, -5.0);

        // onlyPositiveEdits = true
        LearningEditProposal proposal = hc.getBestEdit(true, true);
        assertNull(proposal, "Should not propose negative edits when onlyPositive is true");

        // onlyPositiveEdits = false
        proposal = hc.getBestEdit(true, false);
        assertNotNull(proposal, "Should propose negative edit when onlyPositive is false");
        assertEquals(-5.0, ((ScoreEditMotivation) proposal.getMotivation()).getScore(), 0.001);
    }

    @Test
    public void testRemoveAndInvert() {
        // Setup: Link A->B exists
        try {
            probNet.addLink(varA, varB, true);
        } catch (Exception e) {
            fail("Failed to setup link");
        }

        // Remove A->B score = 8.0
        // Invert A->B score = 12.0
        metric.setScore("A", "B", RemoveLinkEdit.class, 8.0);
        metric.setScore("A", "B", InvertLinkEdit.class, 12.0); // Invert is defined from->to of existing link

        LearningEditProposal proposal = hc.getBestEdit(true, true);
        assertNotNull(proposal);
        assertInstanceOf(InvertLinkEdit.class, proposal.getEdit(), "Should prefer Invert (12.0) over Remove (8.0)");
        assertEquals(12.0, ((ScoreEditMotivation) proposal.getMotivation()).getScore(), 0.001);
    }

    @Test
    public void testHistoryAndRepetition() {
        // Setup:
        // 1. Add A->B (10.0)
        // 2. Add B->C (8.0)
        // 3. Add A->C (6.0)
        metric.setScore("A", "B", AddLinkEdit.class, 10.0);
        metric.setScore("B", "C", AddLinkEdit.class, 8.0);
        metric.setScore("A", "C", AddLinkEdit.class, 6.0);

        // First call: Should get A->B
        LearningEditProposal p1 = hc.getNextEdit(true, true);
        assertNotNull(p1);
        assertEquals(10.0, ((ScoreEditMotivation) p1.getMotivation()).getScore(), 0.001);
        assertEquals("B", ((AddLinkEdit) p1.getEdit()).getVariableTo().getName());

        // Second call (without executing p1): Should get B->C (next best)
        // because A->B implies it was "considered" in the history of this sequence
        LearningEditProposal p2 = hc.getNextEdit(true, true);
        assertNotNull(p2);
        assertEquals(8.0, ((ScoreEditMotivation) p2.getMotivation()).getScore(), 0.001);
        assertEquals("C", ((AddLinkEdit) p2.getEdit()).getVariableTo().getName());

        // Third call: A->C
        LearningEditProposal p3 = hc.getNextEdit(true, true);
        assertNotNull(p3);
        assertEquals(6.0, ((ScoreEditMotivation) p3.getMotivation()).getScore(), 0.001);

        // Reset history (via getBestEdit) -> Start over
        LearningEditProposal pReset = hc.getBestEdit(true, true);
        assertNotNull(pReset);
        assertEquals(10.0, ((ScoreEditMotivation) pReset.getMotivation()).getScore(), 0.001,
                "Should return A->B again after reset");
    }

    // --- Helper Metric Stub ---
    private static class StubMetric extends Metric {

        // Map key format: "FromVarName:ToVarName:EditClassSimpleName" -> Score
        private final Map<String, Double> scores = new HashMap<>();

        public void setScore(String from, String to, Class<? extends PNEdit> type, double score) {
            String key = from + ":" + to + ":" + type.getSimpleName();
            scores.put(key, score);
        }

        @Override
        public double getScore(PNEdit edit) {
            if (edit instanceof BaseLinkEdit linkEdit) {
                String from = linkEdit.getVariableFrom().getName();
                String to = linkEdit.getVariableTo().getName();
                String type = edit.getClass().getSimpleName();
                String key = from + ":" + to + ":" + type;

                return scores.getOrDefault(key, Double.NEGATIVE_INFINITY);
            }
            return Double.NEGATIVE_INFINITY;
        }

        @Override
        public double score(TablePotential nodePotential) {
            return 0;
        }
    }
}
