package org.openmarkov.learning.algorithm.nbderived.snb.util;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;

public class SelectiveNaiveBayesEditProposal extends LearningEditProposal{

    public SelectiveNaiveBayesEditProposal(PNEdit edit, double score) {
            super(edit, new ScoreEditMotivation(score));
        }


}
