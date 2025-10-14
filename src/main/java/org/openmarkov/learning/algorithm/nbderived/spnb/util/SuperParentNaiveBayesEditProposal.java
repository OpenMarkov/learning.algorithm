package org.openmarkov.learning.algorithm.nbderived.spnb.util;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;

public class SuperParentNaiveBayesEditProposal extends LearningEditProposal{

    public SuperParentNaiveBayesEditProposal(PNEdit edit, double score) {
            super(edit, new ScoreEditMotivation(score));
        }


}
