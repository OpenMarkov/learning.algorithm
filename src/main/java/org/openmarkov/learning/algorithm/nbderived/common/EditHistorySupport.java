package org.openmarkov.learning.algorithm.nbderived.common;

import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Helper class to manage edit history for hill-climbing style algorithms.
 * Uses a {@link Set} for O(1) lookup instead of a List.
 */
public class EditHistorySupport {

    private final Set<BaseLinkEdit> history = new LinkedHashSet<>();

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
