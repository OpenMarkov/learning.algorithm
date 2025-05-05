/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */
package org.openmarkov.learning.algorithm.hillclimbing.util;

import org.openmarkov.core.action.PNEdit;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;

public class HillClimbingEditProposal extends LearningEditProposal {
	public HillClimbingEditProposal(PNEdit edit, double score) {
		super(edit, new ScoreEditMotivation(score));
	}

}
