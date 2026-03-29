package org.openmarkov.learning.algorithm.nbderived.treeaugmentednb;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;

/**
 * Edit proposal for the Tree Augmented NB algorithm.
 */
public class TreeAugmentedNBEditProposal extends LearningEditProposal{

    /**
     * Constructs a TAN edit proposal.
     *
     * @param edit  the proposed network edit
     * @param score the score associated with this edit
     */
    public TreeAugmentedNBEditProposal(PNEdit edit, double score) {
            super(edit, new ScoreEditMotivation(score));
        }


}
