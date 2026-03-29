/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */
package org.openmarkov.learning.algorithm.hillclimbing;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;

/**
 * Edit proposal for the Hill Climbing algorithm, wrapping an edit and its score.
 */
public class HillClimbingEditProposal extends LearningEditProposal {
	/**
	 * Constructs a Hill Climbing edit proposal.
	 *
	 * @param edit  the proposed network edit
	 * @param score the score improvement associated with this edit
	 */
	public HillClimbingEditProposal(PNEdit edit, double score) {
		super(edit, new ScoreEditMotivation(score));
	}

}
