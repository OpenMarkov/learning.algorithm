package org.openmarkov.learning.algorithm.nbderived.spnb;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ScoreEditMotivation;

/**
 * Edit proposal for the Super Parent Naive Bayes algorithm.
 */
public class SuperParentNaiveBayesEditProposal extends LearningEditProposal{

    /**
     * Constructs a SPNB edit proposal.
     *
     * @param edit  the proposed network edit
     * @param score the score associated with this edit
     */
    public SuperParentNaiveBayesEditProposal(PNEdit edit, double score) {
            super(edit, new ScoreEditMotivation(score));
        }


}
