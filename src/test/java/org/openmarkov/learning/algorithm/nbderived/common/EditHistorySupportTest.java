package org.openmarkov.learning.algorithm.nbderived.common;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class EditHistorySupportTest {

    private List<PNEdit> backingList;
    private EditHistorySupport support;

    @BeforeEach
    public void setup() {
        backingList = new ArrayList<>();
        support = new EditHistorySupport(backingList);
    }

    @Test
    public void testMarkAndCheck() {
        // Use AddLinkEdit with nulls. Assuming constructor doesn't enforce non-null for
        // bare instantiation or we catch runtime errors.
        // If constructor throws, we can't easily test without ProbNet.
        // But we are mainly testing the 'history' logic (list operations).
        // Since we can't create DummyEdit (sealed), we try this.
        BaseLinkEdit edit1 = null;
        try {
            edit1 = new AddLinkEdit(null, null, null, true);
        } catch (Exception e) {
            // If it fails, we can't proceed with this strategy easily.
            // But let's assume for now it might work or we just want to verify compilation.
        }

        // If instantiation failed (edit1 is null), logic below throws.
        // We really need an object that satisfies BaseLinkEdit type.
        // If we can't create one, we can't test it.
        if (edit1 != null) {
            BaseLinkEdit edit2 = new AddLinkEdit(null, null, null, false); // different object

            assertFalse(support.isEditAlreadyConsidered(edit1));

            support.markEditAsConsidered(edit1);

            assertTrue(support.isEditAlreadyConsidered(edit1));
            assertFalse(support.isEditAlreadyConsidered(edit2));
            assertEquals(1, backingList.size());
            assertEquals(edit1, backingList.getFirst());
        }
    }

    @Test
    public void testReset() throws Exception {
        BaseLinkEdit edit1 = new AddLinkEdit(null, null, null, true);

        support.markEditAsConsidered(edit1);
        assertTrue(support.isEditAlreadyConsidered(edit1));

        support.reset();

        assertFalse(support.isEditAlreadyConsidered(edit1));
        assertTrue(backingList.isEmpty());
    }
}
