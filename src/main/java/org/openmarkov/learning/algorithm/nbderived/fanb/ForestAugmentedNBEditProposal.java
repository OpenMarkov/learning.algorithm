package org.openmarkov.learning.algorithm.nbderived.fanb;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;

public class ForestAugmentedNBEditProposal extends LearningEditProposal{

    public ForestAugmentedNBEditProposal(PNEdit edit, double score) {
            super(edit, new ScoreEditMotivation(score));
        }


}
