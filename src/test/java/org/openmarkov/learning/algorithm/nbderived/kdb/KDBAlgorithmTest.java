package org.openmarkov.learning.algorithm.nbderived.kdb;

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
import org.openmarkov.core.exception.DoEditException;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.potential.TablePotential;
import org.openmarkov.learning.core.util.LearningEditMotivation;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;
import org.openmarkov.learning.metric.Metric;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link KDBAlgorithm}.
 *
 * <p>Uses a stub metric to control scores returned for each edit,
 * verifying KDB's two-phase proposal logic: first class-to-feature
 * edits (scored by unconditioned metric), then feature-to-feature
 * edits (scored by conditioned metric) up to K parents per feature.
 *
 * @author Manuel Arias
 */
public class KDBAlgorithmTest {

    private ProbNet probNet;
    private Variable classVar;
    private Variable f1, f2, f3;
    private CaseDatabase caseDatabase;
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

        // Minimal database so that edit execution doesn't NPE on Metric.initCache
        caseDatabase = new CaseDatabase(
                List.of(classVar, f1, f2, f3),
                new int[][]{{0, 0, 0, 0}, {1, 1, 1, 1}});

        conditioned = new StubMetric();
        unconditioned = new StubMetric();
    }

    private KDBAlgorithm createAlgorithm(int kDependence) {
        KDBAlgorithm kdb = new KDBAlgorithm(probNet, caseDatabase, conditioned, unconditioned, 1.0, kDependence);
        kdb.setClassVariableName("Class");
        return kdb;
    }

    // --- init ---

    @Test
    public void testInitCreatesStarStructureAndConstraints() {
        KDBAlgorithm kdb = createAlgorithm(1);
        kdb.init(null);

        // Class should have 3 children (directed links to all features)
        Node root = probNet.getNode(classVar);
        assertEquals(3, root.getChildren().size(), "Class should have 3 children after init");

        // Constraints should be added
        assertTrue(probNet.hasConstraintOfClass(NoCycle.class), "NoCycle constraint should be present");
        assertTrue(probNet.hasConstraintOfClass(MaxNumParents.class), "MaxNumParents constraint should be present");

        MaxNumParents constraint = probNet.getConstraintOfClass(MaxNumParents.class);
        assertNotNull(constraint);
        assertEquals(2, constraint.getMaxNumParents(),
                "MaxNumParents should be kDependence + 1 = 2");
    }

    @Test
    public void testInitWithK2_MaxNumParentsIs3() {
        KDBAlgorithm kdb = createAlgorithm(2);
        kdb.init(null);

        MaxNumParents constraint = probNet.getConstraintOfClass(MaxNumParents.class);
        assertNotNull(constraint);
        assertEquals(3, constraint.getMaxNumParents(),
                "MaxNumParents should be kDependence + 1 = 3");
    }

    // --- K clamping ---

    @Test
    public void testKDependenceClampedToNumNodesMinus2() {
        // 4 nodes total (1 class + 3 features), max useful K = 2
        // Setting K=10 should clamp to 2
        KDBAlgorithm kdb = createAlgorithm(10);
        kdb.init(null);

        MaxNumParents constraint = probNet.getConstraintOfClass(MaxNumParents.class);
        assertNotNull(constraint);
        // Clamped K = min(10, 4-2) = 2, so MaxNumParents = 3
        assertEquals(3, constraint.getMaxNumParents(),
                "K should be clamped to numNodes - 2 = 2, giving MaxNumParents = 3");
    }

    // --- buildMaximumWeightSpanningTree is no-op ---

    @Test
    public void testBuildMaximumWeightSpanningTreeIsNoOp() {
        unconditioned.setScore("Class", "F1", 5.0);
        unconditioned.setScore("Class", "F2", 10.0);
        unconditioned.setScore("Class", "F3", 3.0);

        KDBAlgorithm kdb = createAlgorithm(1);
        kdb.init(null);

        // KDB overrides buildMaximumWeightSpanningTree with an empty body
        // (it uses its own feature-ordering logic instead). The algorithm
        // should still propose edits without needing a spanning tree.
        LearningEditProposal proposal = kdb.getBestEdit(false, false);
        assertNotNull(proposal, "KDB should propose edits without needing a spanning tree");
    }

    // --- getOptimalEdit: first call returns class-to-feature ---

    @Test
    public void testFirstEditIsClassToFeature() {
        // unconditioned metric: Class->F2 scores highest
        unconditioned.setScore("Class", "F1", 5.0);
        unconditioned.setScore("Class", "F2", 10.0);
        unconditioned.setScore("Class", "F3", 3.0);

        KDBAlgorithm kdb = createAlgorithm(1);
        kdb.init(null);

        LearningEditProposal proposal = kdb.getBestEdit(false, false);
        assertNotNull(proposal, "First edit should not be null");
        assertInstanceOf(AddLinkEdit.class, proposal.getEdit());

        AddLinkEdit edit = (AddLinkEdit) proposal.getEdit();
        assertEquals("Class", edit.getVariableFrom().getName(),
                "First edit should be from Class");
        assertEquals("F2", edit.getVariableTo().getName(),
                "First edit should go to feature with highest unconditioned score (F2)");
    }

    // --- Sequence of edits with K=1 ---

    @Test
    public void testEditSequenceWithK1() throws DoEditException {
        // unconditioned metric: Class->F2 (10), Class->F1 (5), Class->F3 (3)
        unconditioned.setScore("Class", "F2", 10.0);
        unconditioned.setScore("Class", "F1", 5.0);
        unconditioned.setScore("Class", "F3", 3.0);
        // conditioned metric: feature-to-feature scores
        conditioned.setScore("F2", "F1", 7.0);
        conditioned.setScore("F2", "F3", 4.0);
        conditioned.setScore("F1", "F3", 2.0);

        KDBAlgorithm kdb = createAlgorithm(1);
        kdb.init(null);

        // Edit 1: Class->F2 (best unconditioned, adds F2 to domainFeatures)
        // Note: init() already created all Class->Feature links, so these
        // class-to-feature edits serve as ordering signals; the run() loop
        // catches the duplicate-link exception on execution.
        LearningEditProposal p1 = kdb.getBestEdit(false, false);
        assertNotNull(p1);
        AddLinkEdit e1 = (AddLinkEdit) p1.getEdit();
        assertEquals("Class", e1.getVariableFrom().getName());
        assertEquals("F2", e1.getVariableTo().getName());

        // Edit 2: Class->F1 (next best unconditioned, adds F1 to domainFeatures)
        // domainFeatures=["F2"], F2 needs min(1, 0)=0 extra parents → no pending
        LearningEditProposal p2 = kdb.getBestEdit(false, false);
        assertNotNull(p2);
        AddLinkEdit e2 = (AddLinkEdit) p2.getEdit();
        assertEquals("Class", e2.getVariableFrom().getName());
        assertEquals("F1", e2.getVariableTo().getName());

        // Edit 3: F2->F1 (feature-to-feature, K=1 parent needed for F1)
        // domainFeatures=["F2","F1"], F1 has 1 parent (Class), needs min(1,1)=1 extra
        LearningEditProposal p3 = kdb.getBestEdit(false, false);
        assertNotNull(p3);
        AddLinkEdit e3 = (AddLinkEdit) p3.getEdit();
        assertEquals("F2", e3.getVariableFrom().getName(),
                "Feature-to-feature: best domain feature for F1 is F2");
        assertEquals("F1", e3.getVariableTo().getName());

        // Execute the feature-to-feature edit (this one doesn't exist yet)
        e3.executeEdit();

        // Edit 4: Class->F3 (adds F3 to domainFeatures)
        // F1 now has 2 parents, no more pending → next feature
        LearningEditProposal p4 = kdb.getBestEdit(false, false);
        assertNotNull(p4);
        AddLinkEdit e4 = (AddLinkEdit) p4.getEdit();
        assertEquals("Class", e4.getVariableFrom().getName());
        assertEquals("F3", e4.getVariableTo().getName());

        // Edit 5: best feature-to-feature for F3
        // domainFeatures=["F2","F1","F3"], F3 has 1 parent, needs min(1,2)=1 extra
        LearningEditProposal p5 = kdb.getBestEdit(false, false);
        assertNotNull(p5);
        AddLinkEdit e5 = (AddLinkEdit) p5.getEdit();
        assertEquals("F3", e5.getVariableTo().getName());
        assertEquals("F2", e5.getVariableFrom().getName(),
                "Best conditioned score for F3 is F2 (4.0 > F1's 2.0)");
    }

    // --- getMotivation ---

    @Test
    public void testGetMotivationUsesUnconditionedForClassToFeature() {
        unconditioned.setScore("Class", "F1", 8.0);
        conditioned.setScore("Class", "F1", 3.0);

        KDBAlgorithm kdb = createAlgorithm(1);
        kdb.init(null);

        AddLinkEdit edit = new AddLinkEdit(probNet, classVar, f1, true);
        LearningEditMotivation motivation = kdb.getMotivation(edit);

        assertInstanceOf(ScoreEditMotivation.class, motivation);
        assertEquals(8.0, ((ScoreEditMotivation) motivation).getScore(), 0.001,
                "Class->Feature edits should use unconditioned metric");
    }

    @Test
    public void testGetMotivationUsesConditionedForFeatureToFeature() {
        conditioned.setScore("F1", "F2", 6.0);
        unconditioned.setScore("F1", "F2", 1.0);

        KDBAlgorithm kdb = createAlgorithm(1);
        kdb.init(null);

        AddLinkEdit edit = new AddLinkEdit(probNet, f1, f2, true);
        LearningEditMotivation motivation = kdb.getMotivation(edit);

        assertInstanceOf(ScoreEditMotivation.class, motivation);
        assertEquals(6.0, ((ScoreEditMotivation) motivation).getScore(), 0.001,
                "Feature->Feature edits should use conditioned metric");
    }

    // --- Edit history ---

    @Test
    public void testGetBestEditResetsEditHistory() {
        // Set up so that after two class-to-feature selections,
        // a feature-to-feature edit is proposed.
        unconditioned.setScore("Class", "F1", 5.0);
        unconditioned.setScore("Class", "F2", 10.0);
        unconditioned.setScore("Class", "F3", 3.0);
        conditioned.setScore("F2", "F1", 7.0);

        KDBAlgorithm kdb = createAlgorithm(1);
        kdb.init(null);

        // Advance to F2→F1 (feature-to-feature edit)
        kdb.getBestEdit(false, false); // Class→F2
        kdb.getBestEdit(false, false); // Class→F1
        LearningEditProposal p3 = kdb.getBestEdit(false, false); // F2→F1
        assertNotNull(p3);
        assertEquals("F2", ((AddLinkEdit) p3.getEdit()).getVariableFrom().getName());
        assertEquals("F1", ((AddLinkEdit) p3.getEdit()).getVariableTo().getName());

        // Without executing F2→F1, call getBestEdit again.
        // The history was reset, so F2→F1 is no longer marked as considered.
        // F1 still has pending arcs (not executed), so it re-proposes F2→F1.
        LearningEditProposal p4 = kdb.getBestEdit(false, false);
        assertNotNull(p4);
        assertEquals("F2", ((AddLinkEdit) p4.getEdit()).getVariableFrom().getName());
        assertEquals("F1", ((AddLinkEdit) p4.getEdit()).getVariableTo().getName(),
                "After resetHistory, the same feature-to-feature edit should be re-proposed");
    }

    @Test
    public void testDomainFeaturesAdvancesAcrossCalls() {
        unconditioned.setScore("Class", "F1", 5.0);
        unconditioned.setScore("Class", "F2", 10.0);
        unconditioned.setScore("Class", "F3", 3.0);

        KDBAlgorithm kdb = createAlgorithm(1);
        kdb.init(null);

        // First call adds F2 to domainFeatures
        LearningEditProposal p1 = kdb.getBestEdit(false, false);
        assertEquals("F2", ((AddLinkEdit) p1.getEdit()).getVariableTo().getName());

        // Second call: domainFeatures already has F2, so picks F1 next
        LearningEditProposal p2 = kdb.getBestEdit(false, false);
        assertEquals("F1", ((AddLinkEdit) p2.getEdit()).getVariableTo().getName(),
                "domainFeatures persists: second call should pick next best feature");
    }

    // --- Only positive edits ---

    @Test
    public void testOnlyPositiveEditsFilters() {
        // All unconditioned scores are negative
        unconditioned.setScore("Class", "F1", -1.0);
        unconditioned.setScore("Class", "F2", -2.0);
        unconditioned.setScore("Class", "F3", -3.0);

        KDBAlgorithm kdb = createAlgorithm(1);
        kdb.init(null);

        // getNewXMaxFeatureEdit only selects edits with score > 0 (maxScore starts at 0.0)
        LearningEditProposal proposal = kdb.getBestEdit(false, true);
        assertNull(proposal,
                "Should return null when all unconditioned scores are non-positive");
    }

    // --- Stub metric ---

    /**
     * Stub metric that returns configurable scores keyed by "from:to".
     * Overrides getScore(PNEdit) to avoid needing a real CaseDatabase.
     */
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
