package org.openmarkov.learning.algorithm.nbderived.fanb;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;

/**
 * Edit proposal for the Forest Augmented NB algorithm.
 */
public class ForestAugmentedNBEditProposal extends LearningEditProposal{

    /**
     * Constructs a FANB edit proposal.
     *
     * @param edit  the proposed network edit
     * @param score the score associated with this edit
     */
    public ForestAugmentedNBEditProposal(PNEdit edit, double score) {
            super(edit, new ScoreEditMotivation(score));
        }


}
