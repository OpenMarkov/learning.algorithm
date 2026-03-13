package org.openmarkov.learning.algorithm.nbderived.treeaugmentednb;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;

public class TreeAugmentedNBEditProposal extends LearningEditProposal{

    public TreeAugmentedNBEditProposal(PNEdit edit, double score) {
            super(edit, new ScoreEditMotivation(score));
        }


}
