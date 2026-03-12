package org.openmarkov.learning.algorithm.nbderived.common;

import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;

import java.util.List;

/**
 * Helper class to manage edit history for hill-climbing style algorithms
 * in naive bayes derived learners.
 * <p>
 * Extracts duplicated logic found in KDBAlgorithm, TreeAugmentedNBAlgorithm,
 * etc.
 */
public class EditHistorySupport {

    private final List<PNEdit> history;

    /**
     * @param history The list to use for storing edits. typically a protected field
     *                from the algorithm class.
     */
    public EditHistorySupport(List<PNEdit> history) {
        this.history = history;
    }

    public void reset() {
        history.clear();
    }

    public void markEditAsConsidered(BaseLinkEdit edit) {
        history.add(edit);
    }

    public boolean isEditAlreadyConsidered(BaseLinkEdit edit) {
        return history.contains(edit);
    }
}
