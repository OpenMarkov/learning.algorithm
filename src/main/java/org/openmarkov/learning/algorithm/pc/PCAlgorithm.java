/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.pc;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.openmarkov.core.action.*;
import org.openmarkov.core.exception.*;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.graph.Link;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.learning.algorithm.pc.independencetester.IndependenceTester;
import org.openmarkov.learning.algorithm.pc.util.PCEditMotivation;
import org.openmarkov.learning.core.algorithm.LearningAlgorithmType;
import org.openmarkov.learning.core.util.LearningEditMotivation;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.StringEditMotivation;

import javax.swing.event.UndoableEditEvent;
import javax.swing.undo.UndoableEdit;
import java.util.*;

/**
 * PC (Peter-Clark) Algorithm for learning Bayesian Network structure.
 * This algorithm uses conditional independence tests to discover
 * the causal structure of a Bayesian Network.<p>
 * The algorithm works in three main phases:
 * <ol>
 * <li>INITIAL_PHASE: Discovering independence relations and removing links</li>
 * <li>HEAD_TO_HEAD_ORIENTATION: Orienting colliders (head-to-head links)</li>
 * <li>REMAINING_LINKS_ORIENTATION: Orienting remaining links to maintain DAG structure</li>
 * </ol>
 */
@LearningAlgorithmType(name = "PC") 
public class PCAlgorithm extends IndependenceRelationsAlgorithm
		implements PNUndoableEditListener {

	// Constants
	private static final int ALREADY_DONE = -1;

	// Algorithm phases
	private enum Phase {
		INITIAL_PHASE,
		HEAD_TO_HEAD_ORIENTATION,
		REMAINING_LINKS_ORIENTATION,
		ORIENTATION_FINISHED
	}

	// Core algorithm components
	/** Cache for storing independence test results between nodes */
	protected final Map<Node, Map<Node, PCEditMotivation>> cache = new HashMap<>();

	/** History of last best edits returned. */
	protected final Set<PNEdit> lastRemovedEdits = new HashSet<>();
	
	/** History of last best edits returned. */
	protected final Set<PNEdit> lastOrientationEdits = new HashSet<>();
	
	/** History of last best edits returned. */
	protected final List<COrientLinksEdit> lastCompoundOrientationEdits = new ArrayList<>();

	protected IndependenceTester independenceTester;
	
	/** Degree of accuracy of the independence test. */
	protected double significanceLevel;
	
	/** Current algorithm phase */
	private Phase phase;

	/**
	 * Constructor for the PC Algorithm
	 *
	 * @param probNet Probabilistic Network to learn, initially it contains only the nodes.
	 * @param caseDatabase Database of cases
	 * @param alpha Learning rate
	 * @param independenceTester Independence test method
	 * @param significanceLevel Statistical significance level
	 */
	public PCAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Double alpha, IndependenceTester independenceTester,
			Double significanceLevel) {
		super(probNet, caseDatabase, alpha);
		this.independenceTester = independenceTester;
		this.significanceLevel = significanceLevel;
		this.probNet.getPNESupport().addUndoableEditListener(this);

		// Initialize cache for each node
		probNet.getNodes().forEach(node -> cache.put(node, new HashMap<>()));

		// Set initial phase
		this.phase = Phase.INITIAL_PHASE;
	}

	/**
	 * Method that returns the best edit in each step of the algorithm or null
	 * if there are no more edits to consider.
	 *
	 * @param onlyAllowedEdits
	 * @param onlyPositiveEdits
	 * @return LearningEditProposal, or null if no edits are available.
	 */
	@Override 
	public LearningEditProposal getBestEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
		resetHistory();
		return getNextEdit(onlyAllowedEdits, onlyPositiveEdits);
	}

	/**
	 * Method that returns the next best edit in each step of the algorithm
	 * or null if there are no more edits to consider (depending on the
	 * arguments it receives).
	 * @param onlyAllowedEdits if true, only allowed edits are considered
	 * @param onlyPositiveEdits if true, only positive edits are considered
	 * @return LearningEditProposal
	 */
	@Override 
	public LearningEditProposal getNextEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {

		LearningEditProposal bestEditProposal;
		do {
			bestEditProposal = getOptimalEdit(onlyAllowedEdits, onlyPositiveEdits);
		} while (bestEditProposal != null && isBlocked(bestEditProposal)); // Skip blocked edits
		return bestEditProposal;
	}

	/** 
	 * Finds the optimal edit based on current algorithm phase 
	 * and adjacency size.
	 * @param onlyAllowedEdits if true, only allowed edits are considered
	 * @param onlyPositiveEdits if true, only positive edits are considered
	 * @return LearningEditProposal
	 */
	public LearningEditProposal getOptimalEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
		int adjacencySize = 0;
		LearningEditProposal bestEditProposal = null;

		try {
			while (maxOfAdjacencies() > adjacencySize) {
				bestEditProposal = findBestEditInCurrentPhase(adjacencySize, onlyAllowedEdits, onlyPositiveEdits);
				if (bestEditProposal != null) {
					return bestEditProposal;
				}
				adjacencySize++;
			}
		} catch (NodeNotFoundException e) {
			LogManager.getLogger(PCAlgorithm.class.getName()).log(Level.WARN, e);
		}

		// bestEditProposal == null. Transition between phases
		return transitionToNextPhase(onlyAllowedEdits);
	}

	/**
	 * Find the best edit by evaluating separation sets for node pairs.
	 * This method iterates through all nodes and their neighbors,
	 * calculating the best separation set for each pair.
	 */
	private LearningEditProposal findBestEditInCurrentPhase(int adjacencySize,
															boolean onlyAllowedEdits,
															boolean onlyPositiveEdits) throws NodeNotFoundException {
		for (Node nodeX : probNet.getNodes()) {
			for (Node nodeY : nodeX.getSiblings()) {
				List<Node> adjacencySubset = new ArrayList<>(nodeX.getNeighbors());
				adjacencySubset.remove(nodeY);

				RemoveLinkEdit removeLinkEdit = new RemoveLinkEdit(probNet, nodeX.getVariable(),
						nodeY.getVariable(), false);

				if (!alreadyConsidered(removeLinkEdit, lastRemovedEdits)) {
					PCEditMotivation motivation = cache.get(nodeX).get(nodeY);

					// Evaluate separation sets if not already cached or needs recalculation
					if (motivation == null || (motivation.getScore() != ALREADY_DONE
							&& motivation.getSeparationSet().size() > adjacencySize)) {
						evaluateSeparationSets(nodeX, nodeY, adjacencySubset, adjacencySize, onlyPositiveEdits);
					}
				}
			}
		}
		return getOptimalEditFromCache(onlyAllowedEdits, onlyPositiveEdits);
	}

	/**
	 * Transition to next phase when no more edits are possible
	 */
	private LearningEditProposal transitionToNextPhase(boolean onlyAllowedEdits) {
		if (lastRemovedEdits.isEmpty()) {
			phase = Phase.HEAD_TO_HEAD_ORIENTATION;
			return getOrientationEdit(onlyAllowedEdits);
		}
		phase = Phase.INITIAL_PHASE;
		return null;
	}

	 private void evaluateSeparationSets(Node nodeX, Node nodeY, List<Node> adjacencySubset, int adjacencySize, boolean onlyPositiveEdits) throws NodeNotFoundException{
		double bestScore = 0.0;
		List<Node> bestScoreSeparationSet = null;

		for (List<Node> separationSet : subSetsOfSize(adjacencySubset, adjacencySize)) {
			double linkScore = independenceTester.test(caseDatabase, nodeX, nodeY, separationSet);
			if (linkScore > bestScore && (!onlyPositiveEdits || linkScore > significanceLevel)) {
				bestScore = linkScore;
				bestScoreSeparationSet = separationSet;
			}
		}

		if (bestScoreSeparationSet != null) {
			cache.get(nodeX).put(nodeY, new PCEditMotivation(bestScore, bestScoreSeparationSet));
		}
	}

	public LearningEditProposal getOptimalEditFromCache(boolean onlyAllowedEdits, boolean onlyPositiveEdits)
			throws NodeNotFoundException {
		PCEditMotivation bestMotivation = null;
		LearningEditProposal bestEditProposal = null;

		for (Node nodeX : probNet.getNodes()) {
			for (Node nodeY : nodeX.getSiblings()) {
				PCEditMotivation motivation = cache.get(nodeX).get(nodeY);
				if ((motivation != null) && (motivation.getScore() != ALREADY_DONE) && (
						motivation.compareTo(bestMotivation) > 0
				) && (!onlyPositiveEdits || motivation.getScore() > significanceLevel)) {
					RemoveLinkEdit removeLinkEdit = new RemoveLinkEdit(probNet, nodeX.getVariable(),
							nodeY.getVariable(), false);
					if (isValidEdit(removeLinkEdit, bestMotivation, onlyAllowedEdits)) {
						bestMotivation = motivation;
						bestEditProposal = new LearningEditProposal(removeLinkEdit, motivation);
					}
				}
			}
		}
		if (bestEditProposal != null) {
			lastRemovedEdits.add(bestEditProposal.getEdit());
		}
		return bestEditProposal;
	}

	private boolean isValidEdit(RemoveLinkEdit removeLinkEdit, PCEditMotivation bestMotivation, boolean onlyAllowedEdits) {
		return !isBlocked(new LearningEditProposal(removeLinkEdit, bestMotivation)) &&
				!alreadyConsidered(removeLinkEdit, lastRemovedEdits) &&
				(!onlyAllowedEdits || isAllowed(removeLinkEdit));
	}

	/**
	 * Returns the <code>PCEditProposal</code> with the
	 * <code>DirectLinkEdit</code> depending on which stage is the algorithm.
	 * If the "head to head" orientations have not been done, then, the
	 * DirectLinkEdit contains these edits. Else, it contains the remaining
	 * orientations.
	 * @return the orientation edit
	 */
	public LearningEditProposal getOrientationEdit(boolean onlyAllowedEdits) {
		LearningEditProposal bestEdit = null;

		try {
			bestEdit = orientHeadToHeadLinks(onlyAllowedEdits);

			if (bestEdit == null) {
				if (lastCompoundOrientationEdits.isEmpty()) {
					phase = Phase.REMAINING_LINKS_ORIENTATION;
					return orientRemainingLinks(onlyAllowedEdits);
				}
			}
		} catch (NodeNotFoundException | NonProjectablePotentialException | WrongCriterionException e) {
			e.printStackTrace();
		}

		return bestEdit;
	}

	/**
	 * @return the node with the maximum number of neighbors in the probNet.
	 */
	private int maxOfAdjacencies() {
		int max = 0;
		for (Node node : probNet.getNodes()) {
			int adjacents = node.getNumNeighbors();
			if (adjacents > max)
				max = adjacents;
		}
		return max;
	}

	/**
	 * Returns a list of the subsets of size n of the given set
	 *
	 * @param set         <code>List</code> of
	 *                    <code>Node</code> from which extract the subsets.
	 * @param subSetsSize size of the subsets.
	 * @return <code>List</code> of <code>List</code> of
	 * <code>Node</code>. Each <code>List</code> of <code>Node</code>
	 * is one of the subsets of size n.
	 */
	public List<List<Node>> subSetsOfSize(List<Node> set, int subSetsSize) {

		List<List<Node>> subSets = new ArrayList<>();
		List<Node> subSet = new ArrayList<>();
		boolean found = true;
		int[] indexSubSet = new int[subSetsSize];

		//Add the empty set
		if (subSetsSize == 0) {
			subSets.add(new ArrayList<>());
		}

		if ((subSetsSize > 0) && (subSetsSize <= set.size())) {
			for (int i = 0; i < subSetsSize; i++) {
				indexSubSet[i] = i;
				subSet.add(set.get(i));
			}
			subSets.add(subSet);

			if (subSetsSize < set.size()) {
				while (found) {
					found = false;

					for (int i = subSetsSize - 1; i >= 0; i--) {
						if (indexSubSet[i] < (set.size() + (i - subSetsSize))) {
							indexSubSet[i] = indexSubSet[i] + 1;

							if (i < (subSetsSize - 1)) {
								for (int j = i + 1; j < subSetsSize; j++) {
									indexSubSet[j] = indexSubSet[j - 1] + 1;
								}
							}

							found = true;
							break;
						}
					}

					if (found) {
						subSet = new ArrayList<>();
						for (int k = 0; k < subSetsSize; k++) {
							subSet.add(set.get(indexSubSet[k]));
						}

						subSets.add(subSet);
					}
				}
			}
		}

		return subSets;
	}

	/**
	 * Given a RemoveLinkEdit, this method returns the same link with the inverse
	 * direction. For example, if the parameter edit is a RemoveLinkEdit A-&gt;B,
	 * it returns the RemoveLinkEdit B-&gt;A
	 */
	public RemoveLinkEdit inverseEdit(RemoveLinkEdit edit) {
		return new RemoveLinkEdit(probNet, edit.getVariable2(), edit.getVariable1(), false);
	}

	public boolean alreadyConsidered(BaseLinkEdit edit, Set<PNEdit> consideredEdits) {
		BaseLinkEdit inverseEdit = new RemoveLinkEdit(probNet, edit.getVariable2(), edit.getVariable1(),
				edit.isDirected());
		return consideredEdits.contains(edit) || consideredEdits.contains(inverseEdit);
	}

	public boolean alreadyConsidered(OrientLinkEdit edit1, OrientLinkEdit edit2) {
		boolean result = false;

		for (COrientLinksEdit compoundDirectLinkEdit : lastCompoundOrientationEdits) {
			try {
				result |= (
						(edit1.compareTo((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(0)) == 0) && (
								edit2.compareTo((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(1)) == 0
						)
				);
				result |= (
						(edit1.compareTo((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(1)) == 0) && (
								edit2.compareTo((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(0)) == 0
						)
				);
			} catch (NonProjectablePotentialException | WrongCriterionException e) {
				e.printStackTrace();
			}
		}
		return result;
	}

	/**
	 * Method to compute the first stage of the orientation. For each
	 * uncoupled meeting X - Y - Z if Y does not pertain to the separation
	 * set of X and Z, we should orient X -&gt; Y &lt;- Z.
	 */
	private LearningEditProposal orientHeadToHeadLinks(boolean onlyAllowedEdits) throws NodeNotFoundException {

		List<Node> neighborhoodX, neighborhoodY;
		COrientLinksEdit compoundDirectLinkEdit;
		OrientLinkEdit orientLinkEdit1, orientLinkEdit2;
		StringEditMotivation motivation;

		for (Node nodeX : probNet.getNodes()) {
			neighborhoodX = nodeX.getSiblings();
			for (Node nodeY : neighborhoodX) {
				neighborhoodY = nodeY.getSiblings();
				neighborhoodY.remove(nodeX);

				for (Node nodeZ : neighborhoodY) {
					//Adjacent nodeX and nodeZ?
					if (!nodeX.getNeighbors().contains(nodeZ)) {
						// if Y is not included in the separation set of X and Z
						List<Node> separationXZ = cache.get(nodeX).get(nodeZ).getSeparationSet();
						if (!separationXZ.contains(nodeY)) {
							//Then orient X->;Y<-Z
							orientLinkEdit1 = new OrientLinkEdit(probNet, nodeX.getVariable(), nodeY.getVariable(),
									true);
							orientLinkEdit2 = new OrientLinkEdit(probNet, nodeZ.getVariable(), nodeY.getVariable(),
									true);
							compoundDirectLinkEdit = new COrientLinksEdit(probNet, new Vector<>());
							compoundDirectLinkEdit.addEdit(orientLinkEdit1);
							compoundDirectLinkEdit.addEdit(orientLinkEdit2);
							motivation = new StringEditMotivation(
									"Sep. set (" + nodeX.getName() + ", " + nodeZ.getName()
											+ ") does not contain variable: " + nodeY.getName());
							if (!alreadyConsidered(orientLinkEdit1, orientLinkEdit2) && !isBlocked(
									new LearningEditProposal(compoundDirectLinkEdit, motivation)) && (
									!onlyAllowedEdits || (
											isOrientationAllowed(orientLinkEdit1) && isOrientationAllowed(
													orientLinkEdit2)
									)
							)) {
								lastCompoundOrientationEdits.add(compoundDirectLinkEdit);
								return new LearningEditProposal(compoundDirectLinkEdit, motivation);
							}
						}
					}
				}
			}
		}
		return null;
	}

	/**
	 * Method to compute the final stage of the algorithm. The basic idea is
	 * that no new head-to-head links are created and that the DAG condition is
	 * preserved.
	 *
	 * @throws WrongCriterionException
	 * @throws NonProjectablePotentialException
	 */
	private LearningEditProposal orientRemainingLinks(boolean onlyAllowedEdits)
			throws NodeNotFoundException, NonProjectablePotentialException, WrongCriterionException {
		boolean change = true, change2 = true, oriented, skip;
		Node nodeX, nodeZ;
		List<Node> siblingsNodeZ;
		OrientLinkEdit orientLinkEdit = null;
		LearningEditProposal editProposal;

		while (change2) {
			change2 = false;
			while (change) {
				change = false;
				for (Link<Node> link : probNet.getLinks()) {
					nodeX = link.getNode1();
					nodeZ = link.getNode2();
					if (link.isDirected()) {   // X--&gt;Z
						for (Node nodeY : nodeZ.getSiblings()) {
							orientLinkEdit = new OrientLinkEdit(probNet, nodeZ.getVariable(), nodeY.getVariable(),
									true);
							editProposal = new LearningEditProposal(orientLinkEdit,
									new StringEditMotivation("Do not create cycles"));
							if (!nodeY.getNeighbors().contains(nodeX) && !alreadyConsidered(orientLinkEdit,
									lastOrientationEdits) && !isBlocked(editProposal) && (
									!onlyAllowedEdits || isOrientationAllowed(orientLinkEdit)
							)) {
								lastOrientationEdits.add(orientLinkEdit);
								return (editProposal);
							}
						}
					} else { // X -- Z Non-oriented link
						oriented = false;
						orientLinkEdit = new OrientLinkEdit(probNet, nodeX.getVariable(), nodeZ.getVariable(), true);
						editProposal = new LearningEditProposal(orientLinkEdit,
								new StringEditMotivation("Do not create cycles"));
						if (probNet.existsPath(nodeX, nodeZ, true) && !alreadyConsidered(orientLinkEdit,
								lastOrientationEdits) && !isBlocked(editProposal) && (
								!onlyAllowedEdits || isOrientationAllowed(orientLinkEdit)
						)) {
							/* Never used
							change = true;
							oriented = true;
							 */
							lastOrientationEdits.add(orientLinkEdit);
							return (editProposal);
						}
						orientLinkEdit = new OrientLinkEdit(probNet, nodeZ.getVariable(), nodeX.getVariable(), true);
						editProposal = new LearningEditProposal(orientLinkEdit,
								new StringEditMotivation("Do not create cycles"));
						if ((probNet.existsPath(nodeZ, nodeX, true)) && (!oriented) && !alreadyConsidered(
								orientLinkEdit, lastOrientationEdits) && !isBlocked(editProposal) && (
								!onlyAllowedEdits || isOrientationAllowed(orientLinkEdit)
						)) {
							/* Never used
							change = true;
							oriented = true;
							 */
							lastOrientationEdits.add(orientLinkEdit);
							return (editProposal);
						}
						if (!oriented) {// TODO check. !oriented is always true.
							siblingsNodeZ = nodeZ.getSiblings();
							siblingsNodeZ.remove(nodeX);
							for (Node nodeY : siblingsNodeZ) {
								if (!nodeY.getNeighbors().contains(nodeX)) {
									for (Node nodeW : siblingsNodeZ) {
										if (!nodeY.equals(nodeW)) {
											skip = !nodeX.isParent(nodeW) || probNet.getLink(nodeZ, nodeY, true) != null;
											if (!skip) {
												orientLinkEdit = new OrientLinkEdit(probNet, nodeZ.getVariable(),
														nodeW.getVariable(), true);
												editProposal = new LearningEditProposal(orientLinkEdit,
														new StringEditMotivation("Do not create cycles"));
												if (nodeY.getChildren().contains(nodeW) && !alreadyConsidered(
														orientLinkEdit, lastOrientationEdits) && !isBlocked(
														editProposal) && (
														!onlyAllowedEdits || isOrientationAllowed(orientLinkEdit)
												)) {
													/* Never used
													change = true;
													skip = true;
													 */
													lastOrientationEdits.add(orientLinkEdit);
													return (editProposal);
												}
											}
											if (!skip) {
												orientLinkEdit = new OrientLinkEdit(probNet, nodeZ.getVariable(),
														nodeY.getVariable(), true);
												editProposal = new LearningEditProposal(orientLinkEdit,
														new StringEditMotivation("Do not create cycles"));
												if (nodeW.getChildren().contains(nodeY) && !alreadyConsidered(
														orientLinkEdit, lastOrientationEdits) && !isBlocked(
														editProposal) && (
														!onlyAllowedEdits || isOrientationAllowed(orientLinkEdit)
												)) {
													/*
													change = true;
													skip = true;
													 */
													lastOrientationEdits.add(orientLinkEdit);
													return (editProposal);
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}

			for (Link<Node> link : probNet.getLinks()) {
				nodeX = link.getNode1();
				nodeZ = link.getNode2();
				if (!link.isDirected()) {   // X--Z
					orientLinkEdit = new OrientLinkEdit(probNet, nodeX.getVariable(), nodeZ.getVariable(), true);
					editProposal = new LearningEditProposal(orientLinkEdit,
							new StringEditMotivation("Do not create cycles"));
					if (!probNet.existsPath(nodeZ, nodeX, true) && !alreadyConsidered(orientLinkEdit,
							lastOrientationEdits) && !isBlocked(editProposal) && (
							!onlyAllowedEdits || isOrientationAllowed(orientLinkEdit)
					)) {
						// Never used
						// change2 = true;
						lastOrientationEdits.add(orientLinkEdit);
						return (editProposal);
					} else {
						orientLinkEdit = new OrientLinkEdit(probNet, nodeZ.getVariable(), nodeX.getVariable(), true);
						editProposal = new LearningEditProposal(orientLinkEdit,
								new StringEditMotivation("Do not create cycles"));
						if (!isBlocked(editProposal) && (!onlyAllowedEdits || isOrientationAllowed(orientLinkEdit))
								&& (!alreadyConsidered(orientLinkEdit, lastOrientationEdits))) {
							// Never used
							// change2 = true;
							lastOrientationEdits.add(orientLinkEdit);
							return (editProposal);
						}
					}
				}
			}
			orientLinkEdit = null;
		}
		if ((orientLinkEdit == null) && (lastOrientationEdits.isEmpty())) {
			phase = Phase.ORIENTATION_FINISHED;
		}
		return null;
	}

	private boolean isOrientationAllowed(OrientLinkEdit orientLinkEdit) {
		Node sourceNode = probNet.getNode(orientLinkEdit.getVariable1());
		Node destinationNode = probNet.getNode(orientLinkEdit.getVariable2());
		return (
				!probNet.existsPath(destinationNode, sourceNode, true) && isAllowed(orientLinkEdit)
		);
	}

	public void undoableEditWillHappen(UndoableEditEvent event)
			throws ConstraintViolationException {
	}

	public void undoEditHappened(UndoableEditEvent event) {
		UndoableEdit edit = event.getEdit();
		Node nodeX, nodeY;
		double linkScore;

		try {
			if (edit instanceof RemoveLinkEdit) {
				phase = Phase.INITIAL_PHASE;
				RemoveLinkEdit removeLinkEdit = (RemoveLinkEdit) edit;
				nodeX = probNet.getNode(removeLinkEdit.getVariable1());
				nodeY = probNet.getNode(removeLinkEdit.getVariable2());
				List<Node> separationSet = cache.get(nodeX).get(nodeY).getSeparationSet();
				linkScore = independenceTester.test(caseDatabase, nodeX, nodeY, separationSet);
				cache.get(nodeX).put(nodeY, new PCEditMotivation(linkScore, separationSet));
			} else if (edit instanceof AddLinkEdit) {
				AddLinkEdit addLinkEdit = (AddLinkEdit) edit;
				nodeX = probNet.getNode(addLinkEdit.getVariable1());
				nodeY = probNet.getNode(addLinkEdit.getVariable2());
				probNet.removeLink(nodeX, nodeY, false);
				phase = Phase.INITIAL_PHASE;
			} else if (edit instanceof COrientLinksEdit) {
				phase = Phase.INITIAL_PHASE;
			} else if (edit instanceof OrientLinkEdit) {
				phase = Phase.HEAD_TO_HEAD_ORIENTATION;
			}
			resetHistory();
		} catch (NodeNotFoundException e) {
			e.printStackTrace();
		}
	}

	public void undoableEditHappened(UndoableEditEvent event) {

		UndoableEdit edit = event.getEdit();
		Node nodeX, nodeY;

		if (edit instanceof RemoveLinkEdit) {
			RemoveLinkEdit removeLinkEdit = (RemoveLinkEdit) edit;
			nodeX = probNet.getNode(removeLinkEdit.getVariable1());
			nodeY = probNet.getNode(removeLinkEdit.getVariable2());
			List<Node> separationSet = new ArrayList<>();
			PCEditMotivation cachedScore = cache.get(nodeX).get(nodeY);
			if (cachedScore != null) {
				separationSet = cachedScore.getSeparationSet();
			}
			cache.get(nodeX).put(nodeY, new PCEditMotivation(ALREADY_DONE, separationSet));
			cache.get(nodeY).put(nodeX, new PCEditMotivation(ALREADY_DONE, separationSet));
			// Remove the cached values X node's neighbors that contained Y in
			// the separation set (and vice versa)
			for (Node neighborNode : nodeX.getNeighbors()) {
				PCEditMotivation neighborScore = cache.get(nodeX).get(neighborNode);
				if (neighborScore != null && neighborScore.getScore() != ALREADY_DONE && neighborScore
						.getSeparationSet().contains(nodeY)) {
					cache.get(nodeX).remove(neighborNode);
				}
			}
			for (Node neighborNode : nodeY.getNeighbors()) {
				PCEditMotivation neighborScore = cache.get(nodeY).get(neighborNode);
				if (neighborScore != null && neighborScore.getScore() != ALREADY_DONE && neighborScore
						.getSeparationSet().contains(nodeX)) {
					cache.get(nodeY).remove(neighborNode);
				}
			}

		}
		//An AddLinkEdit can only be done by the user. Just undirect the link
		if (edit instanceof AddLinkEdit) {
			AddLinkEdit addLinkEdit = (AddLinkEdit) edit;
			nodeX = probNet.getNode(addLinkEdit.getVariable1());
			nodeY = probNet.getNode(addLinkEdit.getVariable2());
			probNet.removeLink(nodeX, nodeY, true);
			probNet.addLink(nodeX, nodeY, false);
			phase = Phase.INITIAL_PHASE;
		} else if (edit instanceof COrientLinksEdit) {
		// todo: check if this is correct
		}
		resetHistory();
	}

	public LearningEditMotivation getMotivation(PNEdit edit) {
		Node nodeX, nodeY, nodeZ;
		LearningEditMotivation motivation = null;
		if (edit instanceof RemoveLinkEdit) {
			RemoveLinkEdit removeLinkEdit = (RemoveLinkEdit) edit;
			nodeX = probNet.getNode(removeLinkEdit.getVariable1());
			nodeY = probNet.getNode(removeLinkEdit.getVariable2());
			motivation = cache.get(nodeX).get(nodeY);

		} else if (edit instanceof COrientLinksEdit) {
			try {
				COrientLinksEdit compoundDirectLinkEdit = (COrientLinksEdit) edit;
				nodeX = probNet.getNode(((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(0)).getVariable1());
				nodeZ = probNet.getNode(((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(0)).getVariable2());
				nodeY = probNet.getNode(((OrientLinkEdit) compoundDirectLinkEdit.getEdits().get(1)).getVariable1());
				motivation = new StringEditMotivation(
						"Sep. set (" + nodeX.getName() + ", " + nodeY.getName() + ") does not contain variable: "
								+ nodeZ.getName());
			} catch (NonProjectablePotentialException | WrongCriterionException e) {
				e.printStackTrace();
			}
		}
		if (edit instanceof OrientLinkEdit) {
			motivation = new StringEditMotivation("Do not create cycles");
		}
		return motivation;
	}

	@Override public boolean isLastPhase() {
		return (phase.ordinal() >= Phase.REMAINING_LINKS_ORIENTATION.ordinal());
	}

	protected void resetHistory() {
		lastRemovedEdits.clear();
		lastOrientationEdits.clear();
		lastCompoundOrientationEdits.clear();
	}

}