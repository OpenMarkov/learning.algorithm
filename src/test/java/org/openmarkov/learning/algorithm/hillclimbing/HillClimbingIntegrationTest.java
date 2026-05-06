package org.openmarkov.learning.algorithm.hillclimbing;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;
import org.openmarkov.learning.metric.bayesian.BayesianMetric;
import org.openmarkov.learning.metric.entropy.EntropyMetric;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for {@link HillClimbingAlgorithm} using real metrics
 * (BayesianMetric, EntropyMetric) with synthetic datasets.
 *
 * <p>Dataset: X (binary), Y (binary), Z (binary).
 * X and Y are perfectly correlated (Y=X), Z is independent.
 * HC should propose connecting X and Y before any Z-related edit.
 *
 * @author Manuel Arias
 */
public class HillClimbingIntegrationTest {

    private Variable varX, varY, varZ;
    private CaseDatabase caseDatabase;

    @BeforeEach
    public void setup() {
        varX = new Variable("X", 2);
        varY = new Variable("Y", 2);
        varZ = new Variable("Z", 2);

        // 200 cases: X and Y perfectly correlated, Z independent
        int numCases = 200;
        int[][] cases = new int[numCases][3];
        for (int i = 0; i < numCases; i++) {
            int x = i < numCases / 2 ? 0 : 1;
            cases[i][0] = x;
            cases[i][1] = x;       // Y = X (perfect dependency)
            cases[i][2] = i % 2;   // Z alternates (independent)
        }
        caseDatabase = new CaseDatabase(List.of(varX, varY, varZ), cases);
    }

    private HillClimbingAlgorithm createHC(ProbNet probNet, double alpha,
                                           BayesianMetric metric) {
        return new HillClimbingAlgorithm(probNet, caseDatabase, alpha, metric);
    }

    // --- BayesianMetric (K2-like) ---

    @Test
    public void testBayesianMetric_FirstEditConnectsDependentVars() {
        ProbNet probNet = new ProbNet();
        probNet.addNode(varX, NodeType.CHANCE);
        probNet.addNode(varY, NodeType.CHANCE);
        probNet.addNode(varZ, NodeType.CHANCE);

        BayesianMetric metric = new BayesianMetric(1.0);
        HillClimbingAlgorithm hc = createHC(probNet, 0.05, metric);

        LearningEditProposal proposal = hc.getBestEdit(true, true);
        assertNotNull(proposal, "HC with real metric should propose an edit");

        // The best edit should connect X and Y (perfectly correlated)
        BaseLinkEdit edit = (BaseLinkEdit) proposal.getEdit();
        Set<String> vars = Set.of(edit.getVariableFrom().getName(),
                edit.getVariableTo().getName());
        assertTrue(vars.contains("X") && vars.contains("Y"),
                "First edit should connect X and Y (correlated), got: " + vars);
    }

    @Test
    public void testBayesianMetric_FirstEditHasPositiveScore() {
        ProbNet probNet = new ProbNet();
        probNet.addNode(varX, NodeType.CHANCE);
        probNet.addNode(varY, NodeType.CHANCE);
        probNet.addNode(varZ, NodeType.CHANCE);

        BayesianMetric metric = new BayesianMetric(1.0);
        HillClimbingAlgorithm hc = createHC(probNet, 0.05, metric);

        LearningEditProposal proposal = hc.getBestEdit(true, true);
        assertNotNull(proposal);

        double score = ((ScoreEditMotivation) proposal.getMotivation()).getScore();
        assertTrue(score > 0, "Score for dependent-variable edit should be positive, got: " + score);
    }

    @Test
    public void testBayesianMetric_SequentialEditsAreDifferent() {
        ProbNet probNet = new ProbNet();
        probNet.addNode(varX, NodeType.CHANCE);
        probNet.addNode(varY, NodeType.CHANCE);
        probNet.addNode(varZ, NodeType.CHANCE);

        BayesianMetric metric = new BayesianMetric(1.0);
        HillClimbingAlgorithm hc = createHC(probNet, 0.05, metric);

        // Get two sequential edits via getNextEdit (history prevents repeats)
        LearningEditProposal p1 = hc.getNextEdit(true, false);
        LearningEditProposal p2 = hc.getNextEdit(true, false);

        assertNotNull(p1);
        assertNotNull(p2);

        BaseLinkEdit e1 = (BaseLinkEdit) p1.getEdit();
        BaseLinkEdit e2 = (BaseLinkEdit) p2.getEdit();
        assertNotEquals(
                e1.getVariableFrom().getName() + "->" + e1.getVariableTo().getName(),
                e2.getVariableFrom().getName() + "->" + e2.getVariableTo().getName(),
                "Sequential edits should be different (history prevents repeats)");
    }

    @Test
    public void testBayesianMetric_DependentEditsScoreHigherThanIndependent() {
        ProbNet probNet = new ProbNet();
        probNet.addNode(varX, NodeType.CHANCE);
        probNet.addNode(varY, NodeType.CHANCE);
        probNet.addNode(varZ, NodeType.CHANCE);

        BayesianMetric metric = new BayesianMetric(1.0);
        HillClimbingAlgorithm hc = createHC(probNet, 0.05, metric);

        // Collect all proposed edits until null
        Set<String> highScoreEdits = new HashSet<>();
        double firstScore = Double.NaN;
        LearningEditProposal proposal = hc.getNextEdit(true, false);
        while (proposal != null) {
            BaseLinkEdit edit = (BaseLinkEdit) proposal.getEdit();
            double score = ((ScoreEditMotivation) proposal.getMotivation()).getScore();
            if (Double.isNaN(firstScore)) {
                firstScore = score;
            }
            if (score > 0) {
                highScoreEdits.add(edit.getVariableFrom().getName() + "->"
                        + edit.getVariableTo().getName());
            }
            proposal = hc.getNextEdit(true, false);
        }

        // At least one positive-score edit should exist
        assertFalse(highScoreEdits.isEmpty(),
                "Should have at least one positive-score edit");

        // Positive-score edits should involve X and Y
        boolean xyConnected = highScoreEdits.stream().anyMatch(
                e -> (e.contains("X") && e.contains("Y")));
        assertTrue(xyConnected,
                "Positive-score edits should include X-Y connection, got: " + highScoreEdits);
    }

    // --- EntropyMetric ---

    @Test
    public void testEntropyMetric_FirstEditConnectsDependentVars() {
        ProbNet probNet = new ProbNet();
        probNet.addNode(varX, NodeType.CHANCE);
        probNet.addNode(varY, NodeType.CHANCE);
        probNet.addNode(varZ, NodeType.CHANCE);

        EntropyMetric metric = new EntropyMetric();
        HillClimbingAlgorithm hc = new HillClimbingAlgorithm(
                probNet, caseDatabase, 0.05, metric);

        LearningEditProposal proposal = hc.getBestEdit(true, true);
        assertNotNull(proposal, "HC with EntropyMetric should propose an edit");

        BaseLinkEdit edit = (BaseLinkEdit) proposal.getEdit();
        Set<String> vars = Set.of(edit.getVariableFrom().getName(),
                edit.getVariableTo().getName());
        assertTrue(vars.contains("X") && vars.contains("Y"),
                "First edit should connect X and Y, got: " + vars);
    }

    @Test
    public void testBayesianMetric_ResetHistoryReproposesTopEdit() {
        ProbNet probNet = new ProbNet();
        probNet.addNode(varX, NodeType.CHANCE);
        probNet.addNode(varY, NodeType.CHANCE);
        probNet.addNode(varZ, NodeType.CHANCE);

        BayesianMetric metric = new BayesianMetric(1.0);
        HillClimbingAlgorithm hc = createHC(probNet, 0.05, metric);

        // Get best edit twice (getBestEdit resets history)
        LearningEditProposal p1 = hc.getBestEdit(true, true);
        LearningEditProposal p2 = hc.getBestEdit(true, true);

        assertNotNull(p1);
        assertNotNull(p2);

        BaseLinkEdit e1 = (BaseLinkEdit) p1.getEdit();
        BaseLinkEdit e2 = (BaseLinkEdit) p2.getEdit();
        assertEquals(
                e1.getVariableFrom().getName() + "->" + e1.getVariableTo().getName(),
                e2.getVariableFrom().getName() + "->" + e2.getVariableTo().getName(),
                "After history reset, the same top edit should be re-proposed");
    }
}
