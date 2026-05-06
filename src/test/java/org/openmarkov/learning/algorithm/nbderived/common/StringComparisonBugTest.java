package org.openmarkov.learning.algorithm.nbderived.common;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.potential.TablePotential;
import org.openmarkov.learning.core.util.LearningEditMotivation;
import org.openmarkov.learning.core.util.ScoreEditMotivation;
import org.openmarkov.learning.algorithm.nbderived.fanb.ForestAugmentedNBAlgorithm;
import org.openmarkov.learning.algorithm.nbderived.kdb.KDBAlgorithm;
import org.openmarkov.learning.metric.Metric;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression test for bug #6: getName() comparison in getMotivation().
 *
 * <p>The original code in FANB and KDB used identity comparison ({@code ==})
 * on String values returned by {@code getName()}, which could fail when the
 * String instances are {@code .equals()} but not {@code ==}. The fix changed
 * these to {@code .equals()}.
 *
 * <p>These tests verify that getMotivation correctly dispatches between
 * conditioned and unconditioned metrics based on whether the edit's source
 * variable name equals the class variable name, using value equality.
 *
 * @author Manuel Arias
 */
public class StringComparisonBugTest {

    private ProbNet probNet;
    private Variable classVar, f1, f2;
    private StubMetric conditioned;
    private StubMetric unconditioned;

    @BeforeEach
    public void setup() {
        probNet = new ProbNet();
        classVar = new Variable("Class", 2);
        f1 = new Variable("F1", 2);
        f2 = new Variable("F2", 2);
        probNet.addNode(classVar, NodeType.CHANCE);
        probNet.addNode(f1, NodeType.CHANCE);
        probNet.addNode(f2, NodeType.CHANCE);

        conditioned = new StubMetric();
        unconditioned = new StubMetric();
    }

    // --- FANB ---

    @Test
    public void testFANB_ClassToFeatureUsesUnconditionedMetric() {
        unconditioned.setScore("Class", "F1", 42.0);
        conditioned.setScore("Class", "F1", 7.0);

        ForestAugmentedNBAlgorithm fanb = new ForestAugmentedNBAlgorithm(
                probNet, null, conditioned, unconditioned, 1.0);
        fanb.setClassVariableName("Class");

        AddLinkEdit edit = new AddLinkEdit(probNet, classVar, f1, true);
        LearningEditMotivation motivation = fanb.getMotivation(edit);

        assertEquals(42.0, ((ScoreEditMotivation) motivation).getScore(), 0.001,
                "Class->Feature should use unconditioned metric (value 42), not conditioned (7)");
    }

    @Test
    public void testFANB_FeatureToFeatureUsesConditionedMetric() {
        conditioned.setScore("F1", "F2", 99.0);
        unconditioned.setScore("F1", "F2", 1.0);

        ForestAugmentedNBAlgorithm fanb = new ForestAugmentedNBAlgorithm(
                probNet, null, conditioned, unconditioned, 1.0);
        fanb.setClassVariableName("Class");

        AddLinkEdit edit = new AddLinkEdit(probNet, f1, f2, true);
        LearningEditMotivation motivation = fanb.getMotivation(edit);

        assertEquals(99.0, ((ScoreEditMotivation) motivation).getScore(), 0.001,
                "Feature->Feature should use conditioned metric (99), not unconditioned (1)");
    }

    // --- KDB ---

    @Test
    public void testKDB_ClassToFeatureUsesUnconditionedMetric() {
        unconditioned.setScore("Class", "F1", 55.0);
        conditioned.setScore("Class", "F1", 3.0);

        KDBAlgorithm kdb = new KDBAlgorithm(
                probNet,
                new CaseDatabase(
                        List.of(classVar, f1, f2),
                        new int[][]{{0, 0, 0}, {1, 1, 1}}),
                conditioned, unconditioned, 1.0, 1);
        kdb.setClassVariableName("Class");

        AddLinkEdit edit = new AddLinkEdit(probNet, classVar, f1, true);
        LearningEditMotivation motivation = kdb.getMotivation(edit);

        assertEquals(55.0, ((ScoreEditMotivation) motivation).getScore(), 0.001,
                "Class->Feature should use unconditioned metric (55), not conditioned (3)");
    }

    @Test
    public void testKDB_FeatureToFeatureUsesConditionedMetric() {
        conditioned.setScore("F1", "F2", 77.0);
        unconditioned.setScore("F1", "F2", 2.0);

        KDBAlgorithm kdb = new KDBAlgorithm(
                probNet,
                new CaseDatabase(
                        List.of(classVar, f1, f2),
                        new int[][]{{0, 0, 0}, {1, 1, 1}}),
                conditioned, unconditioned, 1.0, 1);
        kdb.setClassVariableName("Class");

        AddLinkEdit edit = new AddLinkEdit(probNet, f1, f2, true);
        LearningEditMotivation motivation = kdb.getMotivation(edit);

        assertEquals(77.0, ((ScoreEditMotivation) motivation).getScore(), 0.001,
                "Feature->Feature should use conditioned metric (77), not unconditioned (2)");
    }

    // --- Stub ---

    private static class StubMetric extends Metric {
        private final Map<String, Double> scores = new HashMap<>();

        void setScore(String from, String to, double score) {
            scores.put(from + ":" + to, score);
        }

        @Override
        public double getScore(PNEdit edit) {
            if (edit instanceof BaseLinkEdit linkEdit) {
                String key = linkEdit.getVariableFrom().getName()
                        + ":" + linkEdit.getVariableTo().getName();
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
