package org.openmarkov.learning.algorithm.nbderived.kdb;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;

/**
 * Edit proposal for the KDB algorithm, wrapping an edit and its score.
 */
public class KDBEditProposal extends LearningEditProposal{

    /**
     * Constructs a KDB edit proposal.
     *
     * @param edit  the proposed network edit
     * @param score the score associated with this edit
     */
    public KDBEditProposal(PNEdit edit, double score) {
            super(edit, new ScoreEditMotivation(score));
        }


}
