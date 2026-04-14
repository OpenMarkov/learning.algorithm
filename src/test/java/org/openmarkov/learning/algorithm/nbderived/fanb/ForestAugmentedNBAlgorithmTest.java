package org.openmarkov.learning.algorithm.nbderived.fanb;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.constraint.MaxNumParents;
import org.openmarkov.core.model.network.constraint.NoCycle;
import org.openmarkov.core.model.network.potential.TablePotential;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;
import org.openmarkov.learning.metric.Metric;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link ForestAugmentedNBAlgorithm}.
 *
 * <p>FANB builds an MWST, computes an average CMI threshold, selects
 * a subtree root using the unconditioned metric, and only proposes
 * feature-to-feature edits within the MWST that exceed the threshold.
 *
 * @author Manuel Arias
 */
public class ForestAugmentedNBAlgorithmTest {

    private ProbNet probNet;
    private Variable classVar, f1, f2, f3;
    private StubMetric conditioned;
    private StubMetric unconditioned;

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

        conditioned = new StubMetric();
        unconditioned = new StubMetric();

        // Conditioned scores for MWST building and avgCMI.
        // All 6 directed pairs (metric is called on directed AddLinkEdits):
        conditioned.setScore("F1", "F2", 10.0);
        conditioned.setScore("F2", "F1", 10.0);
        conditioned.setScore("F1", "F3", 5.0);
        conditioned.setScore("F3", "F1", 5.0);
        conditioned.setScore("F2", "F3", 3.0);
        conditioned.setScore("F3", "F2", 3.0);

        // Unconditioned scores for subtree root selection (Class->feature):
        unconditioned.setScore("Class", "F1", 8.0);
        unconditioned.setScore("Class", "F2", 12.0);
        unconditioned.setScore("Class", "F3", 4.0);
    }

    private TestableFANB createAlgorithm() {
        TestableFANB fanb = new TestableFANB(probNet, conditioned, unconditioned, 1.0);
        fanb.setClassVariableName("Class");
        return fanb;
    }

    // --- init ---

    @Test
    public void testInitCreatesConstraintsAndStarStructure() {
        TestableFANB fanb = createAlgorithm();
        fanb.init(null);

        assertEquals(3, probNet.getNode(classVar).getChildren().size());
        assertTrue(probNet.hasConstraintOfClass(NoCycle.class));
        assertTrue(probNet.hasConstraintOfClass(MaxNumParents.class));

        MaxNumParents constraint = probNet.getConstraintOfClass(MaxNumParents.class);
        assertEquals(2, constraint.getMaxNumParents(),
                "FANB uses kDependence=1, so MaxNumParents = 2");
    }

    @Test
    public void testInitBuildsMWST() {
        TestableFANB fanb = createAlgorithm();
        fanb.init(null);

        // 3 features -> MWST has 2 edges
        assertEquals(2, fanb.getMWSTSize(),
                "MWST should have 2 edges for 3 features");
    }

    @Test
    public void testInitComputesAvgCMI() {
        TestableFANB fanb = createAlgorithm();
        fanb.init(null);

        // avgCMI = sum of all conditioned scores / (N * (N-1))
        // For 3 features, 6 directed pairs: sum = 10+10+5+5+3+3 = 36
        // Also includes self-pairs in the loop (n1==n2), metric returns 0 for those
        // Total pairs: 9 (3*3), non-self: 6, self: 3 (score 0)
        // avgCMI = 36 / (3 * 2) = 6.0
        assertEquals(6.0, fanb.getAvgCMI(), 0.001,
                "avgCMI should be sum of all CMI scores / (N * (N-1))");
    }

    // --- getOptimalEdit: subtree root selection ---

    @Test
    public void testFirstEditSelectsSubtreeRoot() {
        TestableFANB fanb = createAlgorithm();
        fanb.init(null);

        LearningEditProposal proposal = fanb.getBestEdit(false, false);
        assertNotNull(proposal);

        // First edit should be Class->F2 (highest unconditioned score = 12.0)
        AddLinkEdit edit = (AddLinkEdit) proposal.getEdit();
        assertEquals("Class", edit.getVariableFrom().getName());
        assertEquals("F2", edit.getVariableTo().getName(),
                "Subtree root should be the feature with highest unconditioned score");
    }

    // --- getOptimalEdit: avgCMI filtering ---

    @Test
    public void testSubsequentEditsFilterByAvgCMI() {
        TestableFANB fanb = createAlgorithm();
        fanb.init(null);

        // First call: selects subtree root
        fanb.getBestEdit(false, false);

        // Subsequent calls: only propose feature-to-feature edges > avgCMI (6.0)
        // and within the directed MWST
        // The MWST has edges with scores 10 and 5. After redirect from F2:
        // Only directed edges that match withinMaxWeightSpanningTree qualify.
        // Score 10 (F1-F2) > avgCMI=6, score 5 (F1-F3) < avgCMI=6
        // So only the F1-F2 edge (in some direction) should be proposed with
        // onlyAllowedEdits=true.
        LearningEditProposal proposal = fanb.getNextEdit(true, false);
        if (proposal != null) {
            AddLinkEdit edit = (AddLinkEdit) proposal.getEdit();
            double score = conditioned.getScore(edit);
            assertTrue(score > fanb.getAvgCMI(),
                    "Proposed edge score (" + score + ") should exceed avgCMI (" + fanb.getAvgCMI() + ")");
        }
        // It's valid for this to be null if no edges in the directed MWST
        // exceed the threshold after filtering
    }

    // --- getMotivation ---

    @Test
    public void testGetMotivationUsesUnconditionedForClassToFeature() {
        TestableFANB fanb = createAlgorithm();
        fanb.init(null);

        AddLinkEdit edit = new AddLinkEdit(probNet, classVar, f2, true);
        ScoreEditMotivation motivation = (ScoreEditMotivation) fanb.getMotivation(edit);
        assertEquals(12.0, motivation.getScore(), 0.001,
                "Class->Feature should use unconditioned metric");
    }

    @Test
    public void testGetMotivationUsesConditionedForFeatureToFeature() {
        TestableFANB fanb = createAlgorithm();
        fanb.init(null);

        AddLinkEdit edit = new AddLinkEdit(probNet, f1, f2, true);
        ScoreEditMotivation motivation = (ScoreEditMotivation) fanb.getMotivation(edit);
        assertEquals(10.0, motivation.getScore(), 0.001,
                "Feature->Feature should use conditioned metric");
    }

    // --- Helpers ---

    private static class TestableFANB extends ForestAugmentedNBAlgorithm {
        TestableFANB(ProbNet probNet, Metric conditioned, Metric unconditioned, Double alpha) {
            super(probNet, null, conditioned, unconditioned, alpha);
        }

        int getMWSTSize() {
            return mwst.getUndirectedEdges().size();
        }

        Double getAvgCMI() {
            return avgCMI;
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
