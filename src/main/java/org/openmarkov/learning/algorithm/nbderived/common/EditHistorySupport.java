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

    /**
     * Clears all recorded edit history.
     */
    public void reset() {
        history.clear();
    }

    /**
     * Marks the given edit as already considered.
     *
     * @param edit the edit to mark
     */
    public void markEditAsConsidered(BaseLinkEdit edit) {
        history.add(edit);
    }

    /**
     * Checks whether the given edit has already been considered.
     *
     * @param edit the edit to check
     * @return true if this edit was previously marked as considered
     */
    public boolean isEditAlreadyConsidered(BaseLinkEdit edit) {
        return history.contains(edit);
    }
}
