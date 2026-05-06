package org.openmarkov.learning.algorithm.nbderived.common;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.potential.TablePotential;
import org.openmarkov.learning.core.util.LearningEditMotivation;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ModelNetUse;
import org.openmarkov.learning.core.util.ScoreEditMotivation;
import org.openmarkov.learning.metric.Metric;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link DiscriminativeAlgorithm}.
 *
 * @author Manuel Arias
 */
public class DiscriminativeAlgorithmTest {

    private ProbNet probNet;
    private Variable classVar;
    private Variable f1;
    private Variable f2;
    private Variable f3;
    private StubMetric metric;

    @BeforeEach
    public void setup() {
        probNet = new ProbNet();
        classVar = new Variable("Class");
        f1 = new Variable("F1");
        f2 = new Variable("F2");
        f3 = new Variable("F3");
        probNet.addNode(classVar, NodeType.CHANCE);
        probNet.addNode(f1, NodeType.CHANCE);
        probNet.addNode(f2, NodeType.CHANCE);
        probNet.addNode(f3, NodeType.CHANCE);
        metric = new StubMetric();
    }

    private TestableDiscriminativeAlgorithm createAlgorithm(LearningEditProposal... proposals) {
        TestableDiscriminativeAlgorithm alg = new TestableDiscriminativeAlgorithm(probNet, null, metric, 1.0, proposals);
        alg.setClassVariableName("Class");
        return alg;
    }

    // --- Template method pattern ---

    @Test
    public void testGetBestEditResetsHistoryAndDelegates() {
        AddLinkEdit edit = new AddLinkEdit(probNet, f1, f2, true);
        LearningEditProposal proposal = new LearningEditProposal(edit, new ScoreEditMotivation(5.0));

        TestableDiscriminativeAlgorithm alg = createAlgorithm(proposal);
        alg.markEditAsConsidered(edit);
        assertTrue(alg.isEditAlreadyConsidered(edit));

        // getBestEdit should reset history, then delegate
        LearningEditProposal result = alg.getBestEdit(false, false);
        assertNotNull(result);
        assertSame(proposal, result);
    }

    @Test
    public void testGetNextEditDelegatesToGetOptimalEdit() {
        AddLinkEdit edit = new AddLinkEdit(probNet, f1, f2, true);
        LearningEditProposal proposal = new LearningEditProposal(edit, new ScoreEditMotivation(5.0));

        TestableDiscriminativeAlgorithm alg = createAlgorithm(proposal);
        LearningEditProposal result = alg.getNextEdit(false, false);
        assertSame(proposal, result);
    }

    @Test
    public void testGetNextEditReturnsNullWhenNoEdits() {
        TestableDiscriminativeAlgorithm alg = createAlgorithm();
        assertNull(alg.getNextEdit(false, false));
    }

    // --- getMotivation ---

    @Test
    public void testGetMotivationDelegatesToMetric() {
        metric.fixedScore = 7.5;
        TestableDiscriminativeAlgorithm alg = createAlgorithm();
        AddLinkEdit edit = new AddLinkEdit(probNet, f1, f2, true);

        LearningEditMotivation motivation = alg.getMotivation(edit);
        assertInstanceOf(ScoreEditMotivation.class, motivation);
        assertEquals(7.5, ((ScoreEditMotivation) motivation).getScore(), 0.001);
    }

    // --- Edit history ---

    @Test
    public void testEditHistoryMarkAndCheck() {
        TestableDiscriminativeAlgorithm alg = createAlgorithm();
        AddLinkEdit edit = new AddLinkEdit(probNet, f1, f2, true);

        assertFalse(alg.isEditAlreadyConsidered(edit));
        alg.markEditAsConsidered(edit);
        assertTrue(alg.isEditAlreadyConsidered(edit));
    }

    @Test
    public void testResetHistoryClears() {
        TestableDiscriminativeAlgorithm alg = createAlgorithm();
        AddLinkEdit edit = new AddLinkEdit(probNet, f1, f2, true);

        alg.markEditAsConsidered(edit);
        alg.resetHistory();
        assertFalse(alg.isEditAlreadyConsidered(edit));
    }

    // --- Star structure ---

    @Test
    public void testSetRelationsForRootVariable() {
        TestableDiscriminativeAlgorithm alg = createAlgorithm();
        alg.setRelationsForRootVariable();

        assertEquals(3, alg.getRootNode().getChildren().size());
        for (var feature : alg.getNonRootNodes()) {
            assertTrue(feature.getParents().contains(alg.getRootNode()),
                    feature.getName() + " should have Class as parent");
        }
    }

    // --- getRootNode / getNonRootNodes ---

    @Test
    public void testGetRootNode() {
        TestableDiscriminativeAlgorithm alg = createAlgorithm();
        assertEquals("Class", alg.getRootNode().getVariable().getName());
    }

    @Test
    public void testGetNonRootNodes() {
        TestableDiscriminativeAlgorithm alg = createAlgorithm();
        Set<String> names = alg.getNonRootNodes().stream()
                                .map(n -> n.getVariable().getName())
                                .collect(Collectors.toSet());
        assertEquals(Set.of("F1", "F2", "F3"), names);
    }

    // --- Helpers ---

    /**
     * Minimal concrete subclass that returns pre-configured proposals from getOptimalEdit.
     */
    private static class TestableDiscriminativeAlgorithm extends DiscriminativeAlgorithm {

        private final Queue<LearningEditProposal> proposals;

        TestableDiscriminativeAlgorithm(ProbNet probNet, CaseDatabase caseDatabase,
                                        Metric metric, Double alpha,
                                        LearningEditProposal... proposals) {
            super(probNet, caseDatabase, metric, alpha);
            this.proposals = new LinkedList<>();
            for (LearningEditProposal p : proposals) {
                this.proposals.add(p);
            }
        }

        @Override
        protected LearningEditProposal getOptimalEdit(boolean onlyAllowedEdits,
                                                      boolean onlyPositiveEdits) {
            return proposals.poll();
        }

        @Override
        public void init(ModelNetUse modelNetUse) {
            setRelationsForRootVariable();
        }
    }

    /**
     * Stub metric that returns a fixed score for any edit.
     */
    private static class StubMetric extends Metric {
        double fixedScore = 0.0;

        @Override
        public double getScore(PNEdit edit) {
            return fixedScore;
        }

        @Override
        public double score(TablePotential nodePotential) {
            return 0;
        }
    }
}
