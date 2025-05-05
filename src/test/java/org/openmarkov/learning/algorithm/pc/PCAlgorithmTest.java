/*
 * Copyright (c) CISIAD, UNED, Spain,  2018. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.pc;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.openmarkov.core.exception.NodeNotFoundException;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.potential.TablePotential;
import org.openmarkov.io.database.excel.CSVDataBaseIO;
import org.openmarkov.io.probmodel.reader.PGMXReader_0_2;
import org.openmarkov.learning.algorithm.pc.independencetester.CrossEntropyIndependenceTester;
import org.openmarkov.learning.algorithm.pc.independencetester.IndependenceTester;
import org.openmarkov.learning.core.algorithm.LearningAlgorithm;
import org.openmarkov.learning.core.util.ModelNetUse;

import java.util.Arrays;
import java.util.List;

@TestInstance(TestInstance.Lifecycle.PER_METHOD)
public class PCAlgorithmTest {
	private final double maxError = 1E-6;

	private double alpha = 0.5;

	private IndependenceTester independenceTester;
	private double significanceLevel = 0.05;

	private String learnTestDatabaseFilename = "/learnTestDataBase.dbc";
	private String asiaDatabaseFilename = "/asia10K.csv";
	private String alarmDatabaseFilename = "/alarm500.csv";
	private String alarm10kDatabaseFilename = "/alarm10k.csv";

	@BeforeEach
	public void setUp() throws Exception {
		independenceTester = new CrossEntropyIndependenceTester();
	}

	//@Test
	public void testLearnTestDataBase() throws Exception {
		//TODO Commented and making CaseDatabase = null until fixing Elvira database parser with antlr4
		//ElviraDataBaseIO databaseIO = new ElviraDataBaseIO();
		//CaseDatabase learnTestDatabase = databaseIO.load(getClass().getResource(learnTestDatabaseFilename).getFile());
		CaseDatabase learnTestDatabase = null;
		ProbNet learnedNet = new ProbNet();
		for (Variable variable : learnTestDatabase.getVariables()) {
			learnedNet.addNode(variable, NodeType.CHANCE);
		}

		LearningAlgorithm learningAlgorithm = new PCAlgorithm(learnedNet, learnTestDatabase, alpha, independenceTester,
				significanceLevel);

		double[] probabilities;
		learningAlgorithm.run(new ModelNetUse());
		Node nodeA = learnedNet.getNode("A");
		Node nodeB = learnedNet.getNode("B");
		Node nodeC = learnedNet.getNode("C");
		Node nodeD = learnedNet.getNode("D");
		Node nodeE = learnedNet.getNode("E");
		Node nodeF = learnedNet.getNode("F");
		Variable variableA = nodeA.getVariable();
		Variable variableB = nodeB.getVariable();
		Variable variableC = nodeC.getVariable();
		Variable variableD = nodeD.getVariable();
		//Variable variableE = nodeE.getVariable();

		// check the structure of the learned net
		// present links
		Assertions.assertTrue(nodeC.isParent(nodeA));
		Assertions.assertTrue(nodeD.isParent(nodeA));
		Assertions.assertTrue(nodeC.isParent(nodeB));
		Assertions.assertTrue(nodeD.isParent(nodeC));
		Assertions.assertTrue(nodeE.isParent(nodeD));
		// non-present links
		Assertions.assertFalse(nodeA.isParent(nodeB));
		Assertions.assertFalse(nodeA.isParent(nodeC));
		Assertions.assertFalse(nodeA.isParent(nodeD));
		Assertions.assertFalse(nodeA.isParent(nodeE));
		Assertions.assertFalse(nodeA.isParent(nodeF));
		Assertions.assertFalse(nodeB.isParent(nodeA));
		Assertions.assertFalse(nodeB.isParent(nodeC));
		Assertions.assertFalse(nodeB.isParent(nodeD));
		Assertions.assertFalse(nodeB.isParent(nodeE));
		Assertions.assertFalse(nodeB.isParent(nodeF));
		Assertions.assertFalse(nodeC.isParent(nodeD));
		Assertions.assertFalse(nodeC.isParent(nodeE));
		Assertions.assertFalse(nodeC.isParent(nodeF));
		Assertions.assertFalse(nodeD.isParent(nodeB));
		Assertions.assertFalse(nodeD.isParent(nodeE));
		Assertions.assertFalse(nodeD.isParent(nodeF));
		Assertions.assertFalse(nodeE.isParent(nodeA));
		Assertions.assertFalse(nodeE.isParent(nodeB));
		Assertions.assertFalse(nodeE.isParent(nodeC));
		Assertions.assertFalse(nodeE.isParent(nodeF));
		Assertions.assertFalse(nodeF.isParent(nodeA));
		Assertions.assertFalse(nodeF.isParent(nodeB));
		Assertions.assertFalse(nodeF.isParent(nodeC));
		Assertions.assertFalse(nodeF.isParent(nodeD));
		Assertions.assertFalse(nodeF.isParent(nodeE));
		// chek the CPTs
		// A
		probabilities = ((TablePotential) nodeA.getPotentials().get(0)).getValues();
		Assertions.assertEquals(0.305194, probabilities[0], maxError);
		Assertions.assertEquals(0.694805, probabilities[1], maxError);
		//B
		probabilities = ((TablePotential) nodeB.getPotentials().get(0)).getValues();
		Assertions.assertEquals(0.602897, probabilities[0], maxError);
		Assertions.assertEquals(0.397102, probabilities[1], maxError);
		//C | B, A
		TablePotential cGivenBAPotential = (TablePotential) nodeC.getPotentials().get(0);
		List<Variable> cGivenBAVariables = Arrays.asList(variableC, variableB, variableA);
		cGivenBAPotential = (TablePotential) cGivenBAPotential.reorder(cGivenBAVariables);
		probabilities = cGivenBAPotential.getValues();

		Assertions.assertEquals(0.286111, probabilities[0], maxError);
		Assertions.assertEquals(0.713888, probabilities[1], maxError);
		Assertions.assertEquals(0.5, probabilities[2], maxError);
		Assertions.assertEquals(0.5, probabilities[3], maxError);
		Assertions.assertEquals(0.791764, probabilities[4], maxError);
		Assertions.assertEquals(0.208235, probabilities[5], maxError);
		Assertions.assertEquals(0.402573, probabilities[6], maxError);
		Assertions.assertEquals(0.597426, probabilities[7], maxError);

		//D | C, A
		TablePotential dGivenCAPotential = (TablePotential) nodeD.getPotentials().get(0);
		List<Variable> dGivenCAVariables = Arrays.asList(variableD, variableC, variableA);
		dGivenCAPotential = (TablePotential) dGivenCAPotential.reorder(dGivenCAVariables);
		probabilities = dGivenCAPotential.getValues();

		Assertions.assertEquals(0.491304, probabilities[0], maxError);
		Assertions.assertEquals(0.5086956, probabilities[1], maxError);
		Assertions.assertEquals(0.1536458, probabilities[2], maxError);
		Assertions.assertEquals(0.8463541, probabilities[3], maxError);
		Assertions.assertEquals(0.7343049, probabilities[4], maxError);
		Assertions.assertEquals(0.2656950, probabilities[5], maxError);
		Assertions.assertEquals(0.8585657, probabilities[6], maxError);
		Assertions.assertEquals(0.1414342, probabilities[7], maxError);
	}

	@Test public void testAsia10k() throws Exception {
		CSVDataBaseIO csvReader = new CSVDataBaseIO();
		CaseDatabase asiaDatabase = csvReader.load(getClass().getResource(asiaDatabaseFilename).getFile());
		ProbNet learnedNet = new ProbNet();
		for (Variable variable : asiaDatabase.getVariables()) {
			learnedNet.addNode(variable, NodeType.CHANCE);
		}

		LearningAlgorithm learningAlgorithm = new PCAlgorithm(learnedNet, asiaDatabase, alpha, independenceTester,
				significanceLevel);

		learningAlgorithm.run(new ModelNetUse());

		Node nodeAsia = learnedNet.getNode("VisitToAsia");
		Node nodeTuberculosis = learnedNet.getNode("Tuberculosis");
		Node nodeLungCancer = learnedNet.getNode("LungCancer");
		Node nodeTuberculosisOrCancer = learnedNet.getNode("TuberculosisOrCancer");
		Node nodeBronchitis = learnedNet.getNode("Bronchitis");
		Node nodeSmoker = learnedNet.getNode("Smoker");
		Node nodeXRay = learnedNet.getNode("X-ray");
		Node nodeDyspnea = learnedNet.getNode("Dyspnea");

		Assertions.assertNotNull(nodeAsia);
		Assertions.assertNotNull(nodeTuberculosis);
		Assertions.assertNotNull(nodeLungCancer);
		Assertions.assertNotNull(nodeTuberculosisOrCancer);
		Assertions.assertNotNull(nodeBronchitis);
		Assertions.assertNotNull(nodeSmoker);
		Assertions.assertNotNull(nodeXRay);
		Assertions.assertNotNull(nodeDyspnea);

		// Node asia
		Assertions.assertTrue(nodeTuberculosis.isParent(nodeAsia));
		Assertions.assertFalse(nodeLungCancer.isParent(nodeAsia));
		Assertions.assertFalse(nodeTuberculosisOrCancer.isParent(nodeAsia));
		Assertions.assertFalse(nodeBronchitis.isParent(nodeAsia));
		Assertions.assertFalse(nodeSmoker.isParent(nodeAsia));
		Assertions.assertFalse(nodeXRay.isParent(nodeAsia));
		Assertions.assertFalse(nodeDyspnea.isParent(nodeAsia));

		// Node LungCancer
		Assertions.assertFalse(nodeAsia.isParent(nodeLungCancer));
		Assertions.assertFalse(nodeTuberculosis.isParent(nodeLungCancer));
		Assertions.assertTrue(nodeTuberculosisOrCancer.isParent(nodeLungCancer));
		Assertions.assertFalse(nodeBronchitis.isParent(nodeLungCancer));
		Assertions.assertFalse(nodeSmoker.isParent(nodeLungCancer));
		Assertions.assertFalse(nodeXRay.isParent(nodeLungCancer));
		Assertions.assertFalse(nodeDyspnea.isParent(nodeLungCancer));

		// Node Tuberculosis
		Assertions.assertFalse(nodeAsia.isParent(nodeTuberculosis));
		Assertions.assertFalse(nodeLungCancer.isParent(nodeTuberculosis));
		Assertions.assertTrue(nodeTuberculosisOrCancer.isParent(nodeTuberculosis));
		Assertions.assertFalse(nodeBronchitis.isParent(nodeTuberculosis));
		Assertions.assertFalse(nodeSmoker.isParent(nodeTuberculosis));
		Assertions.assertFalse(nodeXRay.isParent(nodeTuberculosis));
		Assertions.assertFalse(nodeDyspnea.isParent(nodeTuberculosis));

		// Node TuberculosisOrCancer
		Assertions.assertFalse(nodeAsia.isParent(nodeTuberculosisOrCancer));
		Assertions.assertFalse(nodeLungCancer.isParent(nodeTuberculosisOrCancer));
		Assertions.assertFalse(nodeTuberculosis.isParent(nodeTuberculosisOrCancer));
		Assertions.assertFalse(nodeBronchitis.isParent(nodeTuberculosisOrCancer));
		Assertions.assertFalse(nodeSmoker.isParent(nodeTuberculosisOrCancer));
		Assertions.assertTrue(nodeXRay.isParent(nodeTuberculosisOrCancer));
		Assertions.assertTrue(nodeDyspnea.isParent(nodeTuberculosisOrCancer));

		// Node Bronchitis
		Assertions.assertFalse(nodeAsia.isParent(nodeBronchitis));
		Assertions.assertFalse(nodeLungCancer.isParent(nodeBronchitis));
		Assertions.assertFalse(nodeTuberculosis.isParent(nodeBronchitis));
		Assertions.assertFalse(nodeTuberculosisOrCancer.isParent(nodeBronchitis));
		Assertions.assertFalse(nodeBronchitis.isParent(nodeBronchitis));
		Assertions.assertFalse(nodeSmoker.isParent(nodeBronchitis));
		Assertions.assertFalse(nodeXRay.isParent(nodeBronchitis));
		Assertions.assertTrue(nodeDyspnea.isParent(nodeBronchitis));

		// Node Smoker
		Assertions.assertFalse(nodeAsia.isParent(nodeSmoker));
		Assertions.assertTrue(nodeLungCancer.isParent(nodeSmoker));
		Assertions.assertFalse(nodeTuberculosis.isParent(nodeSmoker));
		Assertions.assertFalse(nodeTuberculosisOrCancer.isParent(nodeSmoker));
		Assertions.assertTrue(nodeBronchitis.isParent(nodeSmoker));
		Assertions.assertFalse(nodeXRay.isParent(nodeSmoker));
		Assertions.assertFalse(nodeDyspnea.isParent(nodeSmoker));

		// Node X-ray
		Assertions.assertFalse(nodeAsia.isParent(nodeXRay));
		Assertions.assertFalse(nodeLungCancer.isParent(nodeXRay));
		Assertions.assertFalse(nodeTuberculosis.isParent(nodeXRay));
		Assertions.assertFalse(nodeTuberculosisOrCancer.isParent(nodeXRay));
		Assertions.assertFalse(nodeBronchitis.isParent(nodeXRay));
		Assertions.assertFalse(nodeSmoker.isParent(nodeXRay));
		Assertions.assertFalse(nodeDyspnea.isParent(nodeXRay));

		// Node X-ray
		Assertions.assertFalse(nodeAsia.isParent(nodeDyspnea));
		Assertions.assertFalse(nodeLungCancer.isParent(nodeDyspnea));
		Assertions.assertFalse(nodeTuberculosis.isParent(nodeDyspnea));
		Assertions.assertFalse(nodeTuberculosisOrCancer.isParent(nodeDyspnea));
		Assertions.assertFalse(nodeBronchitis.isParent(nodeDyspnea));
		Assertions.assertFalse(nodeSmoker.isParent(nodeDyspnea));
		Assertions.assertFalse(nodeXRay.isParent(nodeDyspnea));

	}

	@Test public void testAlarm500() throws Exception {

		CSVDataBaseIO csvReader = new CSVDataBaseIO();
		CaseDatabase alarmDatabase = csvReader.load(getClass().getResource(alarmDatabaseFilename).getFile());
		ProbNet learnedNet = new ProbNet();
		for (Variable variable : alarmDatabase.getVariables()) {
			learnedNet.addNode(variable, NodeType.CHANCE);
		}

		LearningAlgorithm learningAlgorithm = new PCAlgorithm(learnedNet, alarmDatabase, alpha, independenceTester,
				significanceLevel);

		learningAlgorithm.run(new ModelNetUse());

		Assertions.assertEquals(34, learnedNet.getLinks().size());
		PGMXReader_0_2 reader = new PGMXReader_0_2();
		ProbNet readNet = reader.loadProbNet(getClass().getResource("/BN-alarm.pgmx").getFile());
		printDifferences(readNet, learnedNet);
	}

	//@Test
	public void testAlarm10k() throws Exception {

		CSVDataBaseIO csvReader = new CSVDataBaseIO();
		CaseDatabase alarm10kDatabase = csvReader.load(getClass().getResource(alarm10kDatabaseFilename).getFile());
		ProbNet learnedNet = new ProbNet();
		for (Variable variable : alarm10kDatabase.getVariables()) {
			learnedNet.addNode(variable, NodeType.CHANCE);
		}

		LearningAlgorithm learningAlgorithm = new PCAlgorithm(learnedNet, alarm10kDatabase, alpha, independenceTester,
				significanceLevel);

		learningAlgorithm.run(new ModelNetUse());

		Assertions.assertEquals(46, learnedNet.getLinks().size());

		PGMXReader_0_2 reader = new PGMXReader_0_2();
		ProbNet readNet = reader.loadProbNet(getClass().getResource("/BN-alarm.pgmx").getFile());
		printDifferences(readNet, learnedNet);
	}

	private void printDifferences(ProbNet originalNet, ProbNet learnedNet) throws NodeNotFoundException {
		int missingLinkCount = 0;
		for (Node node : originalNet.getNodes()) {
			Node learnedNode = learnedNet.getNode(node.getName());
			List<Node> learnedNeighbors = learnedNode.getNeighbors();
			for (Node neighbor : node.getNeighbors()) {
				if (node.getName().compareTo(neighbor.getName()) < 0) {
					boolean found = false;
					int i = 0;
					while (!found && i < learnedNeighbors.size()) {
						found = learnedNeighbors.get(i++).getName().equals(neighbor.getName());
					}
					if (!found) {
						System.out.println("Missing: " + node.getName() + " --- " + neighbor.getName());
						missingLinkCount++;
					}
				}
			}
		}

		int addedLinkCount = 0;
		for (Node node : learnedNet.getNodes()) {
			Node readNode = originalNet.getNode(node.getName());
			List<Node> readNeighbors = readNode.getNeighbors();
			for (Node neighbor : node.getNeighbors()) {
				if (node.getName().compareTo(neighbor.getName()) < 0) {
					boolean found = false;
					int i = 0;
					while (!found && i < readNeighbors.size()) {
						found = readNeighbors.get(i++).getName().equals(neighbor.getName());
					}
					if (!found) {
						System.out.println("Added: " + node.getName() + " --- " + neighbor.getName());
						addedLinkCount++;
					}
				}
			}
		}
		System.out.println("Missing: " + missingLinkCount + "; Added: " + addedLinkCount);
	}
}
