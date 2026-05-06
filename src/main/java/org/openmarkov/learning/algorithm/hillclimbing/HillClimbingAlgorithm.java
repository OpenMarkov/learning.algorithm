/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.hillclimbing;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.action.base.linkEdits.InvertLinkEdit;
import org.openmarkov.core.action.base.linkEdits.RemoveLinkEdit;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.constraint.MaxNumParents;
import org.openmarkov.learning.algorithm.nbderived.common.EditHistorySupport;
import org.openmarkov.learning.algorithm.scoreAndSearch.ScoreAndSearchAlgorithm;
import org.openmarkov.learning.metric.Metric;
import org.openmarkov.learning.core.algorithm.LearningAlgorithmType;
import org.openmarkov.learning.core.util.LearningEditMotivation;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ModelNetUse;
import org.openmarkov.learning.core.util.ScoreEditMotivation;

/**
 * This class implements the basic structure of the classic hill climber
 * algorithm.
 *
 * @author joliva
 * @author manuel
 * @author fjdiez
 * @author ibermejo
 * @version 1.1
 * @since OpenMarkov 1.0
 */
@LearningAlgorithmType(name = "Hill climbing", discriminative = false, supportsUnobservedVariables = false)
public class HillClimbingAlgorithm extends ScoreAndSearchAlgorithm {

	/**
	 * Metric used as heuristic
	 */
	protected Metric metric;

	/**
	 * Net we are generating edits for
	 */
	protected ProbNet probNet;

	/**
	 * Tracks edits already considered in the current search cycle.
	 */
	protected final EditHistorySupport editHistory = new EditHistorySupport();

	/**
	 * Maximum number of parents allowed for any node. A value of 0 (default)
	 * means no limit is enforced.
	 */
	protected int maxNumParents = 0;

	// Constructor

	/**
	 * Constructs a Hill Climbing learning algorithm instance.
	 * <p>
	 * The Hill Climbing algorithm is a greedy local search algorithm that
	 * iteratively
	 * selects the edit (add link, remove link, or invert link) with the highest
	 * score
	 * improvement until no further improvements are possible.
	 *
	 * @param probNet      The probabilistic network to learn
	 * @param caseDatabase The database of cases for learning
	 * @param alpha        Significance level parameter (usage depends on metric)
	 * @param metric       The scoring metric used to evaluate edits (e.g., BIC,
	 *                     AIC, K2)
	 */
	public HillClimbingAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Double alpha, Metric metric) {
		super(probNet, caseDatabase, metric, alpha);
		this.probNet = probNet;
		this.metric = metric;
	}

	/**
	 * Constructs a Hill Climbing learning algorithm instance with a maximum
	 * number of parents per node.
	 *
	 * @param probNet       The probabilistic network to learn
	 * @param caseDatabase  The database of cases for learning
	 * @param alpha         Significance level parameter (usage depends on metric)
	 * @param metric        The scoring metric used to evaluate edits
	 * @param maxNumParents Maximum number of parents allowed for any node;
	 *                      a value of 0 means no limit
	 */
	public HillClimbingAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Double alpha, Metric metric,
			int maxNumParents) {
		this(probNet, caseDatabase, alpha, metric);
		this.maxNumParents = maxNumParents;
	}

	@Override
	public void init(ModelNetUse modelNetUse) {
		super.init(modelNetUse);
		if (maxNumParents > 0) {
			probNet.addConstraint(new MaxNumParents(maxNumParents));
		}
	}

	/**
	 * Returns the motivation (score) for a given edit.
	 * <p>
	 * <b>Note:</b> There is an open question about whether the score should come
	 * from the metric (current implementation) or from inference. The metric-based
	 * approach is computationally efficient but may not capture all dependencies.
	 *
	 * @param edit The proposed network edit
	 * @return A ScoreEditMotivation containing the metric score for this edit
	 */
	@Override
	public LearningEditMotivation getMotivation(PNEdit edit) {
		// TODO: Review. Perhaps score should come from inference?
		return new ScoreEditMotivation(metric.getScore(edit));
	}

	/**
	 * This method returns the best edit (and its associated score)
	 * that can be done to the network that is being learnt.
	 *
	 * @param onlyAllowedEdits  If this parameter is true, only those edits
	 *                          that do not provoke a ConstraintViolatedException
	 *                          are returned
	 * @param onlyPositiveEdits If this parameter is true, only those
	 *                          edits with a positive associated score are returned.
	 * @return {@code LearningEditProposal} with the best edit and its score.
	 */
	@Override
	public LearningEditProposal getBestEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
		resetHistory();
		return getNextEdit(onlyAllowedEdits, onlyPositiveEdits);
	}

	/**
	 * This method returns the next best edit (and its associated score)
	 * that can be done to the network that is being learnt.
	 *
	 * @param onlyAllowedEdits  If this parameter is true, only those edits
	 *                          that do not provoke a ConstraintViolatedException
	 *                          are returned
	 * @param onlyPositiveEdits If this parameter is true, only those
	 *                          edits with a positive associated score are returned.
	 * @return {@code LearningEditProposal} with the best edit and its score.
	 */
	@Override
	public LearningEditProposal getNextEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
		LearningEditProposal bestEdit = getOptimalEdit(probNet, onlyAllowedEdits, onlyPositiveEdits);
		while (bestEdit != null && isBlocked(bestEdit.getEdit())) {
			bestEdit = getOptimalEdit(probNet, onlyAllowedEdits, onlyPositiveEdits);
		}
		return bestEdit;
	}

	/**
	 * Resets the edit history so that previously considered edits can be reconsidered.
	 */
	protected void resetHistory() {
		editHistory.reset();
	}

	/**
	 * Marks the given edit as already considered in this iteration.
	 *
	 * @param edit the edit to mark
	 */
	protected void markEditAsConsidered(BaseLinkEdit edit) {
		editHistory.markEditAsConsidered(edit);
	}

	/**
	 * Checks whether the given edit has already been considered in this iteration.
	 *
	 * @param edit the edit to check
	 * @return true if the edit has already been considered
	 */
	protected boolean isEditAlreadyConsidered(BaseLinkEdit edit) {
		return editHistory.isEditAlreadyConsidered(edit);
	}

	/**
	 * Method to obtain the edit with the highest associated score.
	 *
	 * @param learnedNet net to learn.
	 * @return {@code PNEdit} edit with the highest associated score.
	 */
	private LearningEditProposal getOptimalEdit(ProbNet learnedNet, boolean onlyAllowedEdits,
			boolean onlyPositiveEdits) {
		double bestPartialScore = Double.NEGATIVE_INFINITY;
		LearningEditProposal bestEditProposal = null;
		BaseLinkEdit bestEdit = null;
		for (Variable head : learnedNet.getVariables()) {
			for (Variable tail : learnedNet.getVariables()) {
				if (!head.equals(tail)) {
					Node headNode = learnedNet.getNode(head);
					Node tailNode = learnedNet.getNode(tail);

					if (!headNode.isParent(tailNode)) {
						AddLinkEdit addLinkEdit = new AddLinkEdit(learnedNet, tail, head, true);
						double addScore = metric.getScore(addLinkEdit);
						/*
						 * Check whether the score is the best to the moment and
						 * whether this edit has not been already considered
						 */
						if ((addScore > bestPartialScore) && !isEditAlreadyConsidered(addLinkEdit)) {
							if ((!onlyAllowedEdits || isAllowed(addLinkEdit)) && (!onlyPositiveEdits || addScore > 0)
									&& !isBlocked(addLinkEdit)) {
								bestEdit = addLinkEdit;
								bestPartialScore = addScore;
							}
						}
					} else {
						RemoveLinkEdit removeLinkEdit = new RemoveLinkEdit(learnedNet, tail, head, true);
						double removeScore = metric.getScore(removeLinkEdit);
						/*
						 * Check whether the score is the best to the moment and
						 * whether this edit has not been already considered
						 */
						if ((removeScore > bestPartialScore) && !isEditAlreadyConsidered(removeLinkEdit)) {
							if ((!onlyAllowedEdits || isAllowed(removeLinkEdit))
									&& (!onlyPositiveEdits || removeScore > 0) && !isBlocked(removeLinkEdit)) {
								bestEdit = removeLinkEdit;
								bestPartialScore = removeScore;
							}
						}

						InvertLinkEdit invertLinkEdit = new InvertLinkEdit(learnedNet, tail, head, true);
						double invertScore = metric.getScore(invertLinkEdit);
						/*
						 * Check whether the score is the best to the moment and
						 * whether this edit has not been already considered
						 */
						if ((invertScore > bestPartialScore) && !isEditAlreadyConsidered(invertLinkEdit)) {
							if ((!onlyAllowedEdits || isAllowed(invertLinkEdit))
									&& (!onlyPositiveEdits || invertScore > 0) && !isBlocked(invertLinkEdit)) {
								bestEdit = invertLinkEdit;
								bestPartialScore = invertScore;
							}
						}
					}
				}
			}
		}
		if (bestEdit != null) {
			bestEditProposal = LearningEditProposal.scored(bestEdit, bestPartialScore);
			markEditAsConsidered(bestEdit);
		}
		return bestEditProposal;
	}

}
