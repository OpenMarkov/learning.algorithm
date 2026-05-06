package org.openmarkov.learning.algorithm.pc;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.core.COrientLinksEdit;
import org.openmarkov.core.action.base.linkEdits.OrientLinkEdit;
import org.openmarkov.core.action.base.linkEdits.RemoveLinkEdit;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.graph.Link;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.learning.algorithm.pc.independencetester.CrossEntropyIndependenceTester;
import org.openmarkov.learning.core.util.LearningEditProposal;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for the {@link PCAlgorithm} using CSV databases with known structures.
 *
 * <p>This test class verifies the complete behavior of the algorithm, not isolated methods.
 * Each test starts from:
 * <ul>
 *   <li>a database of cases loaded from a CSV resource,</li>
 *   <li>a {@link ProbNet} initialized as a complete undirected graph,</li>
 *   <li>execution of the PC algorithm until it finishes or until the relevant phase is reached.</li>
 * </ul>
 *
 * <p>The goal is to check that the algorithm:
 * <ul>
 *   <li>removes links during the skeleton discovery phase when conditional independence is detected,</li>
 *   <li>detects and orients v-structures (head-to-head patterns),</li>
 *   <li>continues to the remaining orientation phases,</li>
 *   <li>behaves correctly both in direct execution and in GUI-like step-by-step execution.</li>
 * </ul>
 *
 * <p>The tests use two reference networks:
 * <ul>
 *   <li><b>ThreeNodesAtoBtoC</b>: a chain {@code A -> B -> C}. The algorithm should remove
 *   the link between {@code A} and {@code C}, because they become conditionally independent
 *   given {@code B}.</li>
 *   <li><b>HeadToHead1</b>: a collider {@code A -> C <- B}. The algorithm should remove
 *   the link between {@code A} and {@code B}, and later orient the remaining links as
 *   {@code A -> C <- B}.</li>
 * </ul>
 *
 * <p>These tests are especially useful because they validate interactions between phases and
 * internal state transitions that are hard to verify with isolated unit tests.
 */
public class PCAlgorithmIntegrationTest {

    /**
     * Significance level used by the conditional independence test.
     *
     * <p>This value is passed both as the algorithm significance threshold and to the
     * {@link CrossEntropyIndependenceTester}. It determines when two variables are considered
     * conditionally independent from the data.
     */
    private static final double SIGNIFICANCE = 0.05;

// -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /**
     * Loads a CSV file from the test resources and converts it into a {@link CaseDatabase}.
     *
     * <p>The first row is interpreted as the header and provides the variable names.
     * The remaining rows are the observed cases.
     *
     * <p>For each column, state names are discovered dynamically from the data in the order
     * in which they first appear. That order is then used to assign integer state indices
     * required by {@link CaseDatabase}.
     *
     * <p>The list received in {@code variables} is cleared and populated with the
     * {@link Variable} instances created from the CSV header. This allows the caller to use
     * the exact same variables later when building the {@link ProbNet}.
     *
     * @param resourcePath path of the CSV file inside the test resources.
     * @param variables output parameter that will be filled with the variables discovered in
     *                  the CSV header.
     * @return a case database containing the variables and the cases encoded as integer states.
     * @throws Exception if the resource cannot be read or if any parsing problem occurs.
     */

    private static CaseDatabase loadCsvDatabase(String resourcePath, List<Variable> variables) throws Exception {
        InputStream is = PCAlgorithmIntegrationTest.class.getClassLoader().getResourceAsStream(resourcePath);
        assertNotNull(is, "Test resource not found: " + resourcePath);

        List<String[]> rawRows = new ArrayList<>();
        String[] headers;

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
            headers = reader.readLine().split(",");
            for (int i = 0; i < headers.length; i++) {
                headers[i] = headers[i].trim();
            }

            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    rawRows.add(line.split(","));
                }
            }
        }

        // For each column, discover the set of states in first-seen order.
        // LinkedHashMap is used to preserve insertion order deterministically.
        @SuppressWarnings("unchecked")
        LinkedHashMap<String, Integer>[] stateMaps = new LinkedHashMap[headers.length];
        for (int i = 0; i < headers.length; i++) {
            stateMaps[i] = new LinkedHashMap<>();
        }
        for (String[] row : rawRows) {
            for (int i = 0; i < row.length; i++) {
                String state = row[i].trim();
                stateMaps[i].putIfAbsent(state, stateMaps[i].size());
            }
        }

        // Create Variable objects.
        // The number of states of each variable is the number of distinct values
        // seen in its column.
        variables.clear();
        for (int i = 0; i < headers.length; i++) {
            variables.add(new Variable(headers[i], stateMaps[i].size()));
        }

        // Build cases array (string states → integer indices)
        int[][] cases = new int[rawRows.size()][headers.length];
        for (int r = 0; r < rawRows.size(); r++) {
            for (int c = 0; c < headers.length; c++) {
                cases[r][c] = stateMaps[c].get(rawRows.get(r)[c].trim());
            }
        }

        return new CaseDatabase(variables, cases);
    }

    /**
     * Builds a {@link ProbNet} containing the given variables as chance nodes and connects
     * every pair of nodes with an undirected link.
     *
     * <p>This is the standard starting point for the PC algorithm: a complete undirected graph
     * from which links will be removed when independences are discovered.
     *
     * @param variables variables that will become the nodes of the network.
     * @return a probabilistic network initialized as a complete undirected graph.
     */
    private static ProbNet buildCompleteUndirectedGraph(List<Variable> variables) {
        ProbNet probNet = new ProbNet();
        List<Node> nodes = new ArrayList<>();
        for (Variable var : variables) {
            nodes.add(probNet.addNode(var, NodeType.CHANCE));
        }
        for (int i = 0; i < nodes.size(); i++) {
            for (int j = i + 1; j < nodes.size(); j++) {
                probNet.addLink(nodes.get(i), nodes.get(j), false);
            }
        }
        return probNet;
    }

    /**
     * Runs the PC algorithm until no more edits are proposed.
     *
     * <p>At each iteration:
     * <ol>
     *   <li>the best available edit is requested through {@link PCAlgorithm#getBestEdit(boolean, boolean)},</li>
     *   <li>that edit is executed,</li>
     *   <li>the algorithm is notified through {@link PCAlgorithm#afterEditExecutes(PNEdit)}.</li>
     * </ol>
     *
     * <p>The method counts how many removals and orientations were actually executed. This is
     * useful for tests that want to assert not only the final graph but also the type of work
     * performed by the algorithm.
     *
     * <p>A safety limit of 1000 iterations is used to avoid infinite loops in case of a defect.
     *
     * @param pc algorithm instance to execute.
     * @return a map with two counters:
     * <ul>
     *   <li>{@code "removals"}: number of executed {@link RemoveLinkEdit} edits,</li>
     *   <li>{@code "orientations"}: number of executed orientation edits
     *   ({@link COrientLinksEdit} or {@link OrientLinkEdit}).</li>
     * </ul>
     * @throws Exception if the algorithm or an edit fails during execution.
     */
    private static Map<String, Integer> runAlgorithmToCompletion(PCAlgorithm pc) throws Exception {
        int removals = 0;
        int orientations = 0;
        int maxSteps = 1000;

        for (int step = 0; step < maxSteps; step++) {
            LearningEditProposal proposal = pc.getBestEdit(true, true);
            if (proposal == null) {
                break;
            }
            PNEdit edit = proposal.getEdit();
            edit.executeEdit();
            pc.afterEditExecutes(edit);

            if (edit instanceof RemoveLinkEdit) {
                removals++;
            } else if (edit instanceof COrientLinksEdit || edit instanceof OrientLinkEdit) {
                orientations++;
            }
        }

        Map<String, Integer> counts = new HashMap<>();
        counts.put("removals", removals);
        counts.put("orientations", orientations);
        return counts;
    }

    /**
     * Runs the PC algorithm while simulating the GUI behavior of peeking ahead at future edits.
     *
     * <p>This method reproduces what happens in step-by-step mode in the interactive learning
     * dialog: before executing the first proposal shown to the user, the GUI calls
     * {@link PCAlgorithm#getBestEdit(boolean, boolean)} once and then invokes
     * {@link PCAlgorithm#getNextEdit(boolean, boolean)} several times in order to fill the
     * proposals table.
     *
     * <p>Although only the first proposal is executed, those extra {@code getNextEdit()} calls
     * may have internal side effects on the algorithm phase. This helper exists precisely to
     * test that such peeking does not break the intended phase progression.
     *
     * <p>This is particularly relevant for bugs in which the algorithm advances to a later phase
     * while the GUI is only exploring proposals, causing the actual execution path to skip
     * v-structure detection.
     *
     * @param pc algorithm instance to execute.
     * @param peekCount maximum number of calls to {@code getNextEdit()} before each real edit.
     * @return a map with the number of removal and orientation edits actually executed.
     * @throws Exception if the algorithm or an edit fails during execution.
     */
    private static Map<String, Integer> runAlgorithmWithGuiPeeking(PCAlgorithm pc, int peekCount)
            throws Exception {
        int removals = 0;
        int orientations = 0;
        int maxSteps = 1000;

        for (int step = 0; step < maxSteps; step++) {
            // First proposal shown by the GUI.
            LearningEditProposal best = pc.getBestEdit(true, true);
            if (best == null) {
                break;
            }
            // Simulate the GUI filling the proposals table by peeking ahead.
            // These calls are intentionally performed because they may affect
            // the internal phase of the algorithm.
            for (int i = 0; i < peekCount; i++) {
                if (pc.getNextEdit(true, true) == null) {
                    break;
                }
            }

            // In the actual GUI workflow, only the first proposal is executed.
            PNEdit edit = best.getEdit();
            edit.executeEdit();
            pc.afterEditExecutes(edit);

            if (edit instanceof RemoveLinkEdit) {
                removals++;
            } else if (edit instanceof COrientLinksEdit || edit instanceof OrientLinkEdit) {
                orientations++;
            }
        }

        Map<String, Integer> counts = new HashMap<>();
        counts.put("removals", removals);
        counts.put("orientations", orientations);
        return counts;
    }

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------

    /**
     * Verifies the skeleton discovered for the chain {@code A -> B -> C}.
     *
     * <p>In a chain, variables {@code A} and {@code C} become conditionally independent
     * given {@code B}, so the PC algorithm must remove the link between them.
     * However, links {@code A-B} and {@code B-C} must remain.
     *
     * @throws Exception if any test setup or algorithm execution step fails.
     */
    @Test
    @DisplayName("Chain A→B→C: removes only A–C and links A–B and B–C survive")
    void testChain_skeletonDiscovery() throws Exception {
        List<Variable> variables = new ArrayList<>();
        CaseDatabase db = loadCsvDatabase("network/ThreeNodesAtoBtoC.csv", variables);
        ProbNet probNet = buildCompleteUndirectedGraph(variables);

        PCAlgorithm pc = new PCAlgorithm(probNet, db, SIGNIFICANCE,
                new CrossEntropyIndependenceTester(), SIGNIFICANCE, null);

        runAlgorithmToCompletion(pc);

        Node nodeA = probNet.getNode(variables.get(0));
        Node nodeB = probNet.getNode(variables.get(1));
        Node nodeC = probNet.getNode(variables.get(2));

        // A–C must be absent in every form (A is conditionally independent of C given B)
        assertNull(probNet.getLink(nodeA, nodeC, false), "Undirected A–C should be removed");
        assertNull(probNet.getLink(nodeA, nodeC, true),  "Directed A→C should not exist");
        assertNull(probNet.getLink(nodeC, nodeA, true),  "Directed C→A should not exist");

        // A–B must survive (A and B are not independent)
        boolean abExists = probNet.getLink(nodeA, nodeB, false) != null
                || probNet.getLink(nodeA, nodeB, true) != null
                || probNet.getLink(nodeB, nodeA, true) != null;
        assertTrue(abExists, "Link A–B should still exist");

        // B–C must survive (B and C are not independent)
        boolean bcExists = probNet.getLink(nodeB, nodeC, false) != null
                || probNet.getLink(nodeB, nodeC, true) != null
                || probNet.getLink(nodeC, nodeB, true) != null;
        assertTrue(bcExists, "Link B–C should still exist");
    }

    /**
     * Verifies that in the chain {@code A -> B -> C} exactly one link is removed.
     *
     * <p>The only removable link in this structure is {@code A-C}. This test complements the
     * previous one by asserting the number of removal edits, not only the final graph.
     *
     * @throws Exception if any test setup or algorithm execution step fails.
     */
    @Test
    @DisplayName("Chain A→B→C: exactly one link removed (A–C)")
    void testChain_exactlyOneLinkRemoved() throws Exception {
        List<Variable> variables = new ArrayList<>();
        CaseDatabase db = loadCsvDatabase("network/ThreeNodesAtoBtoC.csv", variables);
        ProbNet probNet = buildCompleteUndirectedGraph(variables);

        PCAlgorithm pc = new PCAlgorithm(probNet, db, SIGNIFICANCE,
                new CrossEntropyIndependenceTester(), SIGNIFICANCE, null);

        Map<String, Integer> counts = runAlgorithmToCompletion(pc);

        assertEquals(1, counts.get("removals"), "Exactly one link should be removed in a chain");
    }

    /**
     * Verifies the collider case {@code A -> C <- B}.
     *
     * <p>In a v-structure, {@code A} and {@code B} are marginally independent, so the link
     * between them must be removed. The remaining links must then be oriented toward
     * {@code C}, resulting in {@code A -> C <- B}.
     *
     * @throws Exception if any test setup or algorithm execution step fails.
     */
    @Test
    @DisplayName("Collider A→C←B: removes A–B and orients collider A→C←B")
    void testCollider_vStructureOrientation() throws Exception {
        List<Variable> variables = new ArrayList<>();
        CaseDatabase db = loadCsvDatabase("network/HeadToHead1.csv", variables);
        ProbNet probNet = buildCompleteUndirectedGraph(variables);

        PCAlgorithm pc = new PCAlgorithm(probNet, db, SIGNIFICANCE,
                new CrossEntropyIndependenceTester(), SIGNIFICANCE, null);

        runAlgorithmToCompletion(pc);

        Node nodeA = probNet.getNode(variables.get(0));
        Node nodeB = probNet.getNode(variables.get(1));
        Node nodeC = probNet.getNode(variables.get(2));

        // A–B must be absent (A and B are marginally independent in a v-structure)
        assertNull(probNet.getLink(nodeA, nodeB, false), "Undirected A–B should be removed");
        assertNull(probNet.getLink(nodeA, nodeB, true),  "Directed A→B should not exist");
        assertNull(probNet.getLink(nodeB, nodeA, true),  "Directed B→A should not exist");

        // A→C must be present and directed
        Link<Node> linkAC = probNet.getLink(nodeA, nodeC, true);
        assertNotNull(linkAC, "Directed A→C should exist");

        // B→C must be present and directed
        Link<Node> linkBC = probNet.getLink(nodeB, nodeC, true);
        assertNotNull(linkBC, "Directed B→C should exist");
    }

    /**
     * Verifies the number of edits executed in the collider case.
     *
     * <p>The algorithm should remove exactly one link ({@code A-B}) and perform at least one
     * orientation edit in order to produce the collider.
     *
     * @throws Exception if any test setup or algorithm execution step fails.
     */
    @Test
    @DisplayName("Collider A→C←B: exactly one link removed (A–B) and at least one orientation")
    void testCollider_editCounts() throws Exception {
        List<Variable> variables = new ArrayList<>();
        CaseDatabase db = loadCsvDatabase("network/HeadToHead1.csv", variables);
        ProbNet probNet = buildCompleteUndirectedGraph(variables);

        PCAlgorithm pc = new PCAlgorithm(probNet, db, SIGNIFICANCE,
                new CrossEntropyIndependenceTester(), SIGNIFICANCE, null);

        Map<String, Integer> counts = runAlgorithmToCompletion(pc);

        assertEquals(1, counts.get("removals"), "Exactly one link should be removed");
        assertTrue(counts.get("orientations") >= 1, "At least one orientation edit should be performed");
    }

    /**
     * Verifies that, in step-by-step execution, the algorithm reaches an orientation phase
     * after performing skeleton discovery on the collider example.
     *
     * <p>This test does not assert the exact final graph. Its purpose is narrower:
     * to ensure that the algorithm does not stop prematurely after removals and that
     * orientation edits are actually proposed and executed afterward.
     *
     * @throws Exception if any test setup or algorithm execution step fails.
     */
    @Test
    @DisplayName("Step-by-step: orientation edits are proposed after skeleton phase (HeadToHead1)")
    void testStepByStep_reachesOrientationPhase() throws Exception {
        List<Variable> variables = new ArrayList<>();
        CaseDatabase db = loadCsvDatabase("network/HeadToHead1.csv", variables);
        ProbNet probNet = buildCompleteUndirectedGraph(variables);

        PCAlgorithm pc = new PCAlgorithm(probNet, db, SIGNIFICANCE,
                new CrossEntropyIndependenceTester(), SIGNIFICANCE, null);

        boolean sawRemoval = false;
        boolean sawOrientation = false;
        int maxSteps = 1000;

        for (int step = 0; step < maxSteps; step++) {
            LearningEditProposal proposal = pc.getBestEdit(true, true);
            if (proposal == null) {
                break;
            }
            PNEdit edit = proposal.getEdit();
            edit.executeEdit();
            pc.afterEditExecutes(edit);

            if (edit instanceof RemoveLinkEdit) {
                sawRemoval = true;
            } else if (edit instanceof COrientLinksEdit || edit instanceof OrientLinkEdit) {
                sawOrientation = true;
            }
        }

        assertTrue(sawRemoval, "Expected at least one link removal during skeleton phase");
        assertTrue(sawOrientation,
                "Expected orientation edits: execution must not stop before the orientation phases");
    }

    /**
     * Verifies that, in step-by-step execution, the algorithm also reaches an orientation
     * phase in the chain example.
     *
     * <p>Even though the chain case is structurally simpler than the collider, the execution
     * flow should still continue beyond the removal phase.
     *
     * @throws Exception if any test setup or algorithm execution step fails.
     */
    @Test
    @DisplayName("Step-by-step: orientation edits are proposed after skeleton phase (ThreeNodesAtoBtoC)")
    void testStepByStep_chainReachesOrientationPhase() throws Exception {
        List<Variable> variables = new ArrayList<>();
        CaseDatabase db = loadCsvDatabase("network/ThreeNodesAtoBtoC.csv", variables);
        ProbNet probNet = buildCompleteUndirectedGraph(variables);

        PCAlgorithm pc = new PCAlgorithm(probNet, db, SIGNIFICANCE,
                new CrossEntropyIndependenceTester(), SIGNIFICANCE, null);

        boolean sawRemoval = false;
        boolean sawOrientation = false;
        int maxSteps = 1000;

        for (int step = 0; step < maxSteps; step++) {
            LearningEditProposal proposal = pc.getBestEdit(true, true);
            if (proposal == null) {
                break;
            }
            PNEdit edit = proposal.getEdit();
            edit.executeEdit();
            pc.afterEditExecutes(edit);

            if (edit instanceof RemoveLinkEdit) {
                sawRemoval = true;
            } else if (edit instanceof COrientLinksEdit || edit instanceof OrientLinkEdit) {
                sawOrientation = true;
            }
        }

        assertTrue(sawRemoval, "Expected the A–C link to be removed");
        assertTrue(sawOrientation,
                "Expected orientation edits: execution must not stop before the orientation phases");
    }

    // -------------------------------------------------------------------------
    // Exact-GUI-flow tests (no explicit afterEditExecutes, with parametricLearning)
    // -------------------------------------------------------------------------

    /**
     * Simulates exactly what {@code LearningManager.applyEdit()} does:
     * {@code edit.executeEdit()} (which fires {@code afterEditExecutes} once via PNESupport)
     * followed by {@code parametricLearning()}.  No explicit {@code pc.afterEditExecutes()}
     * call — that is the key difference from the regular test helpers.
     */
    private static Map<String, Integer> runAlgorithmWithExactGuiFlow(PCAlgorithm pc, ProbNet probNet,
                                                                     int peekCount)
            throws Exception {
        probNet.getPNESupport().setWithUndo(true);

        int removals = 0;
        int orientations = 0;
        int maxSteps = 1000;

        for (int step = 0; step < maxSteps; step++) {
            LearningEditProposal best = pc.getBestEdit(true, true);
            if (best == null) {
                break;
            }
            for (int i = 0; i < peekCount; i++) {
                if (pc.getNextEdit(true, true) == null) {
                    break;
                }
            }
            PNEdit edit = best.getEdit();
            edit.executeEdit();           // fires afterEditExecutes via PNESupport
            pc.parametricLearning();      // exactly like LearningManager.applyEdit — NO explicit afterEditExecutes

            if (edit instanceof RemoveLinkEdit) {
                removals++;
            } else if (edit instanceof COrientLinksEdit || edit instanceof OrientLinkEdit) {
                orientations++;
            }
        }

        Map<String, Integer> counts = new HashMap<>();
        counts.put("removals", removals);
        counts.put("orientations", orientations);
        return counts;
    }

    @Test
    @DisplayName("Exact GUI flow — Collider A→C←B: v-structure detected without explicit afterEditExecutes")
    void testExactGuiFlow_collider_vStructureDetected() throws Exception {
        List<Variable> variables = new ArrayList<>();
        CaseDatabase db = loadCsvDatabase("network/HeadToHead1.csv", variables);
        ProbNet probNet = buildCompleteUndirectedGraph(variables);

        PCAlgorithm pc = new PCAlgorithm(probNet, db, SIGNIFICANCE,
                new CrossEntropyIndependenceTester(), SIGNIFICANCE, null);

        Map<String, Integer> counts = runAlgorithmWithExactGuiFlow(pc, probNet, 50);

        Node nodeA = probNet.getNode(variables.get(0));
        Node nodeB = probNet.getNode(variables.get(1));
        Node nodeC = probNet.getNode(variables.get(2));

        assertNull(probNet.getLink(nodeA, nodeB, false), "Undirected A–B should be removed");
        assertNull(probNet.getLink(nodeA, nodeB, true),  "Directed A→B should not exist");
        assertNull(probNet.getLink(nodeB, nodeA, true),  "Directed B→A should not exist");

        assertNotNull(probNet.getLink(nodeA, nodeC, true), "Directed A→C should exist");
        assertNotNull(probNet.getLink(nodeB, nodeC, true), "Directed B→C should exist");

        assertEquals(1, counts.get("removals"),     "Exactly 1 removal expected");
        assertTrue(counts.get("orientations") >= 1, "At least 1 orientation expected");
    }

    @Test
    @DisplayName("Exact GUI flow — Chain A→B→C: skeleton correct without explicit afterEditExecutes")
    void testExactGuiFlow_chain_skeletonAndOrientationCorrect() throws Exception {
        List<Variable> variables = new ArrayList<>();
        CaseDatabase db = loadCsvDatabase("network/ThreeNodesAtoBtoC.csv", variables);
        ProbNet probNet = buildCompleteUndirectedGraph(variables);

        PCAlgorithm pc = new PCAlgorithm(probNet, db, SIGNIFICANCE,
                new CrossEntropyIndependenceTester(), SIGNIFICANCE, null);

        Map<String, Integer> counts = runAlgorithmWithExactGuiFlow(pc, probNet, 50);

        Node nodeA = probNet.getNode(variables.get(0));
        Node nodeB = probNet.getNode(variables.get(1));
        Node nodeC = probNet.getNode(variables.get(2));

        assertNull(probNet.getLink(nodeA, nodeC, false), "Undirected A–C should be removed");

        boolean abExists = probNet.getLink(nodeA, nodeB, false) != null
                || probNet.getLink(nodeA, nodeB, true) != null
                || probNet.getLink(nodeB, nodeA, true) != null;
        assertTrue(abExists, "Link A–B should still exist");

        boolean bcExists = probNet.getLink(nodeB, nodeC, false) != null
                || probNet.getLink(nodeB, nodeC, true) != null
                || probNet.getLink(nodeC, nodeB, true) != null;
        assertTrue(bcExists, "Link B–C should still exist");

        assertEquals(1, counts.get("removals"), "Exactly 1 removal expected");
        assertTrue(counts.get("orientations") >= 1, "Algorithm should reach orientation phase");
    }

    // -------------------------------------------------------------------------
    // GUI-peeking tests (simulate InteractiveLearningDialog.updateEditionsTable)
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("GUI peeking — Collider A→C←B: v-structure detected even when getNextEdit() peeks ahead")
    void testGuiPeeking_collider_vStructureDetected() throws Exception {
        List<Variable> variables = new ArrayList<>();
        CaseDatabase db = loadCsvDatabase("network/HeadToHead1.csv", variables);
        ProbNet probNet = buildCompleteUndirectedGraph(variables);

        PCAlgorithm pc = new PCAlgorithm(probNet, db, SIGNIFICANCE,
                new CrossEntropyIndependenceTester(), SIGNIFICANCE, null);

        // 50 peeks — same as SHOWING_EDIT_NUM in InteractiveLearningDialog
        Map<String, Integer> counts = runAlgorithmWithGuiPeeking(pc, 50);

        Node nodeA = probNet.getNode(variables.get(0));
        Node nodeB = probNet.getNode(variables.get(1));
        Node nodeC = probNet.getNode(variables.get(2));

        // A–B must be removed
        assertNull(probNet.getLink(nodeA, nodeB, false), "Undirected A–B should be removed");
        assertNull(probNet.getLink(nodeA, nodeB, true),  "Directed A→B should not exist");
        assertNull(probNet.getLink(nodeB, nodeA, true),  "Directed B→A should not exist");

        // Collider A→C←B must be oriented via HEAD_TO_HEAD (not arbitrary REMAINING_LINKS)
        assertNotNull(probNet.getLink(nodeA, nodeC, true), "Directed A→C should exist");
        assertNotNull(probNet.getLink(nodeB, nodeC, true), "Directed B→C should exist");

        assertEquals(1, counts.get("removals"),     "Exactly 1 removal expected");
        assertTrue(counts.get("orientations") >= 1, "At least 1 orientation expected");
    }

    // -------------------------------------------------------------------------
    // HeadToHead2 end-to-end test
    // -------------------------------------------------------------------------

    /**
     * Verifies that the PC algorithm recovers the skeleton of the HeadToHead2 network
     * and correctly identifies the v-structure at C from 5000 cases of data.
     *
     * <p>The reference network (HeadToHead2.pgmx) has five variables and the following
     * directed links:
     * <pre>
     *   A → C ← B       (v-structure / collider at C)
     *       B → E
     *   C → F ← E       (v-structure / collider at F)
     * </pre>
     *
     * <p><b>Skeleton (reliably recovered):</b> All five edges are present — A–C, B–C,
     * B–E, C–F, E–F — and none of the five absent pairs (A–B, A–E, A–F, B–F, C–E)
     * have a link.
     *
     * <p><b>V-structure at C (reliably recovered):</b> A and B are marginally
     * independent (empty separation set), so C is always identified as a collider:
     * A→C←B.
     *
     * <p><b>Note on the v-structure at F:</b> With 5000 cases the chi-square test
     * for {@code B ⊥ F | {E}} exceeds the significance threshold, so the algorithm
     * records sep(B,F) = {E}. Because C ∉ {E}, a spurious v-structure is detected at C
     * for the pair (B,F), which causes C–F to be oriented as F→C instead of C→F.
     * For this reason the test does not assert the orientation of C–F; it only checks
     * that C–F exists in some form.
     *
     * @throws Exception if any test setup or algorithm execution step fails.
     */
    @Test
    @DisplayName("HeadToHead2: correct skeleton and v-structure A→C←B recovered from 5000 cases")
    void testHeadToHead2_skeletonAndColliderAtC() throws Exception {
        List<Variable> variables = new ArrayList<>();
        CaseDatabase db = loadCsvDatabase("network/HeadToHead2.csv", variables);
        ProbNet probNet = buildCompleteUndirectedGraph(variables);

        PCAlgorithm pc = new PCAlgorithm(probNet, db, SIGNIFICANCE,
                new CrossEntropyIndependenceTester(), SIGNIFICANCE, null);

        runAlgorithmToCompletion(pc);

        System.out.println("\n=== HeadToHead2 (5k) ===");
        printEdges(probNet);

        // Resolve nodes by variable name to be independent of CSV column order.
        Node nodeA = probNet.getNode("A");
        Node nodeB = probNet.getNode("B");
        Node nodeC = probNet.getNode("C");
        Node nodeE = probNet.getNode("E");
        Node nodeF = probNet.getNode("F");

        // --- Non-adjacent pairs must have no link in any form ---
        assertAbsent(probNet, nodeA, nodeB, "A–B");
        assertAbsent(probNet, nodeA, nodeE, "A–E");
        assertAbsent(probNet, nodeA, nodeF, "A–F");
        assertAbsent(probNet, nodeB, nodeF, "B–F");
        assertAbsent(probNet, nodeC, nodeE, "C–E");

        // --- Adjacent pairs from the reference skeleton must all be present ---
        boolean acExists = probNet.getLink(nodeA, nodeC, false) != null
                || probNet.getLink(nodeA, nodeC, true) != null
                || probNet.getLink(nodeC, nodeA, true) != null;
        assertTrue(acExists, "Edge A–C must be present");

        boolean bcExists = probNet.getLink(nodeB, nodeC, false) != null
                || probNet.getLink(nodeB, nodeC, true) != null
                || probNet.getLink(nodeC, nodeB, true) != null;
        assertTrue(bcExists, "Edge B–C must be present");

        boolean beExists = probNet.getLink(nodeB, nodeE, false) != null
                || probNet.getLink(nodeB, nodeE, true) != null
                || probNet.getLink(nodeE, nodeB, true) != null;
        assertTrue(beExists, "Edge B–E must be present");

        boolean cfExists = probNet.getLink(nodeC, nodeF, false) != null
                || probNet.getLink(nodeC, nodeF, true) != null
                || probNet.getLink(nodeF, nodeC, true) != null;
        assertTrue(cfExists, "Edge C–F must be present");

        boolean efExists = probNet.getLink(nodeE, nodeF, false) != null
                || probNet.getLink(nodeE, nodeF, true) != null
                || probNet.getLink(nodeF, nodeE, true) != null;
        assertTrue(efExists, "Edge E–F must be present");

        // --- V-structure at C: A→C and B→C must be directed toward C ---
        // A and B are marginally independent (empty separation set), so this
        // v-structure is always correctly identified.
        assertNotNull(probNet.getLink(nodeA, nodeC, true),
                "Directed A→C must exist (v-structure at C)");
        assertNotNull(probNet.getLink(nodeB, nodeC, true),
                "Directed B→C must exist (v-structure at C)");
    }


    /**
     * Verifies that the PC algorithm fully recovers the structure of HeadToHead2
     * from 10000 cases of data.
     *
     * <p>With 10000 cases the chi-square test correctly rejects {@code B⊥F|{E}},
     * so no spurious v-structure is created and both colliders are properly oriented:
     * <pre>
     *   A → C ← B       (v-structure at C)
     *       B → E
     *   C → F ← E       (v-structure at F)
     * </pre>
     *
     * <p>The orientation of B–E may be B→E, E→B, or undirected, because no PC
     * orientation rule forces a direction on that edge after both v-structures are
     * resolved.
     *
     * @throws Exception if any test setup or algorithm execution step fails.
     */
    @Test
    @DisplayName("HeadToHead2 (10k): full structure recovered — skeleton, A→C←B and C→F←E")
    void testHeadToHead2_10k_fullStructureRecovered() throws Exception {
        List<Variable> variables = new ArrayList<>();
        CaseDatabase db = loadCsvDatabase("network/HeadToHead2-10k.csv", variables);
        ProbNet probNet = buildCompleteUndirectedGraph(variables);

        PCAlgorithm pc = new PCAlgorithm(probNet, db, SIGNIFICANCE,
                new CrossEntropyIndependenceTester(), SIGNIFICANCE, null);

        runAlgorithmToCompletion(pc);

        Node nodeA = probNet.getNode("A");
        Node nodeB = probNet.getNode("B");
        Node nodeC = probNet.getNode("C");
        Node nodeE = probNet.getNode("E");
        Node nodeF = probNet.getNode("F");

        // --- Non-adjacent pairs must be absent ---
        assertAbsent(probNet, nodeA, nodeB, "A–B");
        assertAbsent(probNet, nodeA, nodeE, "A–E");
        assertAbsent(probNet, nodeA, nodeF, "A–F");
        assertAbsent(probNet, nodeB, nodeF, "B–F");
        assertAbsent(probNet, nodeC, nodeE, "C–E");

        // --- V-structure at C ---
        assertNotNull(probNet.getLink(nodeA, nodeC, true), "Directed A→C must exist");
        assertNotNull(probNet.getLink(nodeB, nodeC, true), "Directed B→C must exist");

        // --- V-structure at F ---
        assertNotNull(probNet.getLink(nodeC, nodeF, true), "Directed C→F must exist");
        assertNotNull(probNet.getLink(nodeE, nodeF, true), "Directed E→F must exist");

        // --- B–E must be present in some form ---
        boolean beExists = probNet.getLink(nodeB, nodeE, false) != null
                || probNet.getLink(nodeB, nodeE, true) != null
                || probNet.getLink(nodeE, nodeB, true) != null;
        assertTrue(beExists, "Edge B–E must be present");
    }

    // -------------------------------------------------------------------------
    // Asia network (8 variables, 10k cases) — comparison with Tetrad
    // -------------------------------------------------------------------------

    /**
     * Runs PC on the Asia Bayesian network (10 000 cases) and verifies the
     * recovered CPDAG against the ground-truth structure.
     *
     * <p>True Asia DAG:
     * <pre>
     *   VisitToAsia → Tuberculosis
     *   Smoker → LungCancer
     *   Smoker → Bronchitis
     *   Tuberculosis → TuberculosisOrCancer ← LungCancer   (v-structure)
     *   TuberculosisOrCancer → X-ray
     *   TuberculosisOrCancer → Dyspnea ← Bronchitis         (v-structure)
     * </pre>
     *
     * <p>The CPDAG must satisfy:
     * <ul>
     *   <li>All 8 skeleton edges are present.</li>
     *   <li>Key absent edges are absent (the two v-structures make several
     *       pairs d-separated by the empty set).</li>
     *   <li>Both v-structures are oriented correctly.</li>
     *   <li>{@code TuberculosisOrCancer → X-ray} is oriented by Meek R1.</li>
     * </ul>
     *
     * <p>The three remaining edges (VisitToAsia—Tuberculosis,
     * Smoker—LungCancer, Smoker—Bronchitis) are reversible in the MEC and
     * may be left undirected; no orientation is asserted for them.
     *
     * <p>This test produces a summary printed to stdout so that the output
     * can be compared directly with the equivalent Tetrad test
     * ({@code TestPcAsiaComparison} in the Tetrad project).
     */
    @Test
    @DisplayName("Asia (10k): skeleton and CPDAG match true structure")
    void testAsia_skeletonAndCpdag() throws Exception {
        List<Variable> variables = new ArrayList<>();
        CaseDatabase db = loadCsvDatabase("network/BN-asia10k.csv", variables);
        ProbNet probNet = buildCompleteUndirectedGraph(variables);

        PCAlgorithm pc = new PCAlgorithm(probNet, db, SIGNIFICANCE,
                new CrossEntropyIndependenceTester(), SIGNIFICANCE, null);

        runAlgorithmToCompletion(pc);

        // Print result for manual comparison with Tetrad
        System.out.println("\n=== OpenMarkov PC - Asia (10k, alpha=" + SIGNIFICANCE + ") ===");
        printEdges(probNet);

        // Resolve nodes by variable name
        Node visit    = probNet.getNode("VisitToAsia");
        Node smoker   = probNet.getNode("Smoker");
        Node lung     = probNet.getNode("LungCancer");
        Node bronch   = probNet.getNode("Bronchitis");
        Node tub      = probNet.getNode("Tuberculosis");
        Node tubOrCa  = probNet.getNode("TuberculosisOrCancer");
        Node xray     = probNet.getNode("X-ray");
        Node dysp     = probNet.getNode("Dyspnea");

        // --- Skeleton: detectable edges must be present in some form ---
        //
        // NOTE: VisitToAsia — Tuberculosis is intentionally NOT asserted.
        // VisitToAsia=yes appears only ~100 times in 10k rows (P≈0.01).
        // The expected cell count for (VisitToAsia=yes, Tuberculosis=yes) under
        // independence is ~0.94 < 1, making the chi-square test unreliable.
        // Both OpenMarkov and Tetrad remove this edge with 10k samples.
        assertPresent(probNet, smoker,  lung,    "Smoker — LungCancer");
        assertPresent(probNet, smoker,  bronch,  "Smoker — Bronchitis");
        assertPresent(probNet, tub,     tubOrCa, "Tuberculosis — TuberculosisOrCancer");
        assertPresent(probNet, lung,    tubOrCa, "LungCancer — TuberculosisOrCancer");
        assertPresent(probNet, tubOrCa, xray,    "TuberculosisOrCancer — X-ray");
        assertPresent(probNet, tubOrCa, dysp,    "TuberculosisOrCancer — Dyspnea");
        assertPresent(probNet, bronch,  dysp,    "Bronchitis — Dyspnea");

        // --- Absent edges: non-adjacent pairs in the true skeleton ---
        // (VisitToAsia pairs are omitted: that node becomes isolated because
        //  its only true edge, VisitToAsia—Tuberculosis, is undetectable with
        //  10k samples — see note above.)
        assertAbsent(probNet, smoker, tub,    "Smoker — Tuberculosis");
        assertAbsent(probNet, lung,   bronch, "LungCancer — Bronchitis");
        assertAbsent(probNet, tub,    bronch, "Tuberculosis — Bronchitis");
        assertAbsent(probNet, xray,   dysp,   "X-ray — Dyspnea");

        // --- V-structure 1: Tuberculosis → TuberculosisOrCancer ← LungCancer ---
        assertNotNull(probNet.getLink(tub,  tubOrCa, true),
                "Tuberculosis → TuberculosisOrCancer must be directed (v-structure 1)");
        assertNotNull(probNet.getLink(lung, tubOrCa, true),
                "LungCancer → TuberculosisOrCancer must be directed (v-structure 1)");

        // --- V-structure 2: TuberculosisOrCancer → Dyspnea ← Bronchitis ---
        assertNotNull(probNet.getLink(tubOrCa, dysp, true),
                "TuberculosisOrCancer → Dyspnea must be directed (v-structure 2)");
        assertNotNull(probNet.getLink(bronch,  dysp, true),
                "Bronchitis → Dyspnea must be directed (v-structure 2)");

        // --- Meek R1: TuberculosisOrCancer → X-ray ---
        // TuberculosisOrCancer has a directed parent (Tuberculosis or LungCancer from v-structure 1),
        // and X-ray is not adjacent to either of them → R1 forces TuberculosisOrCancer → X-ray.
        assertNotNull(probNet.getLink(tubOrCa, xray, true),
                "TuberculosisOrCancer → X-ray must be directed (Meek R1)");
    }

    // -------------------------------------------------------------------------
    // Helpers shared by all tests
    // -------------------------------------------------------------------------

    /**
     * Asserts that an edge exists between {@code n1} and {@code n2} in some form
     * (undirected, n1→n2, or n2→n1).
     */
    private static void assertPresent(ProbNet net, Node n1, Node n2, String label) {
        boolean exists = net.getLink(n1, n2, false) != null
                || net.getLink(n1, n2, true)  != null
                || net.getLink(n2, n1, true)  != null;
        assertTrue(exists, "Edge " + label + " must be present");
    }

    /**
     * Asserts that there is no link between {@code n1} and {@code n2} in any form
     * (undirected, n1→n2, or n2→n1).
     */
    private static void assertAbsent(ProbNet net, Node n1, Node n2, String label) {
        assertNull(net.getLink(n1, n2, false), "Undirected " + label + " should not exist");
        assertNull(net.getLink(n1, n2, true),  "Directed " + label + " should not exist");
        assertNull(net.getLink(n2, n1, true),  "Directed " + label + " (reversed) should not exist");
    }

    /**
     * Prints all edges in the network to stdout in alphabetical order.
     * Undirected edges are shown as "A --- B" and directed as "A --> B".
     */
    private static void printEdges(ProbNet probNet) {
        List<String> lines = new ArrayList<>();
        for (Link<Node> link : probNet.getLinks()) {
            String from = link.getFrom().getName();
            String to   = link.getTo().getName();
            if (link.isDirected()) {
                lines.add(from + " --> " + to);
            } else {
                // Canonical order: lexicographically smaller name first
                String a = from.compareTo(to) <= 0 ? from : to;
                String b = from.compareTo(to) <= 0 ? to   : from;
                lines.add(a + " --- " + b);
            }
        }
        Collections.sort(lines);
        for (String line : lines) {
            System.out.println("  " + line);
        }
    }

    @Test
    @DisplayName("GUI peeking — Chain A→B→C: skeleton and orientation still correct")
    void testGuiPeeking_chain_skeletonAndOrientationCorrect() throws Exception {
        List<Variable> variables = new ArrayList<>();
        CaseDatabase db = loadCsvDatabase("network/ThreeNodesAtoBtoC.csv", variables);
        ProbNet probNet = buildCompleteUndirectedGraph(variables);

        PCAlgorithm pc = new PCAlgorithm(probNet, db, SIGNIFICANCE,
                new CrossEntropyIndependenceTester(), SIGNIFICANCE, null);

        Map<String, Integer> counts = runAlgorithmWithGuiPeeking(pc, 50);

        Node nodeA = probNet.getNode(variables.get(0));
        Node nodeB = probNet.getNode(variables.get(1));
        Node nodeC = probNet.getNode(variables.get(2));

        // A–C must be removed
        assertNull(probNet.getLink(nodeA, nodeC, false), "Undirected A–C should be removed");
        assertNull(probNet.getLink(nodeA, nodeC, true),  "Directed A→C should not exist");
        assertNull(probNet.getLink(nodeC, nodeA, true),  "Directed C→A should not exist");

        // A–B must survive
        boolean abExists = probNet.getLink(nodeA, nodeB, false) != null
                || probNet.getLink(nodeA, nodeB, true) != null
                || probNet.getLink(nodeB, nodeA, true) != null;
        assertTrue(abExists, "Link A–B should still exist");

        // B–C must survive
        boolean bcExists = probNet.getLink(nodeB, nodeC, false) != null
                || probNet.getLink(nodeB, nodeC, true) != null
                || probNet.getLink(nodeC, nodeB, true) != null;
        assertTrue(bcExists, "Link B–C should still exist");

        assertEquals(1, counts.get("removals"), "Exactly 1 removal expected");
        assertTrue(counts.get("orientations") >= 1, "Algorithm should reach orientation phase");
    }
}
