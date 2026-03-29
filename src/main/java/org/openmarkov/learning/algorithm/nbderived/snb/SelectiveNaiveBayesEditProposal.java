package org.openmarkov.learning.algorithm.nbderived.snb;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;

/**
 * Edit proposal for the Selective Naive Bayes algorithm.
 */
public class SelectiveNaiveBayesEditProposal extends LearningEditProposal{

    /**
     * Constructs a SNB edit proposal.
     *
     * @param edit  the proposed network edit
     * @param score the score associated with this edit
     */
    public SelectiveNaiveBayesEditProposal(PNEdit edit, double score) {
            super(edit, new ScoreEditMotivation(score));
        }


}
