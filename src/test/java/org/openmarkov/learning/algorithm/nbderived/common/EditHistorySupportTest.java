package org.openmarkov.learning.algorithm.nbderived.common;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openmarkov.core.action.base.linkEdits.AddLinkEdit;
import org.openmarkov.core.action.base.linkEdits.BaseLinkEdit;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;

import static org.junit.jupiter.api.Assertions.*;

public class EditHistorySupportTest {

    private EditHistorySupport support;
    private ProbNet probNet;
    private Variable varA;
    private Variable varB;

    @BeforeEach
    public void setup() {
        support = new EditHistorySupport();

        probNet = new ProbNet();
        varA = new Variable("A", 2);
        varB = new Variable("B", 2);
        probNet.addNode(varA, NodeType.CHANCE);
        probNet.addNode(varB, NodeType.CHANCE);
    }

    @Test
    public void testMarkAndCheck() {
        BaseLinkEdit edit1 = new AddLinkEdit(probNet, varA, varB, true);
        BaseLinkEdit edit2 = new AddLinkEdit(probNet, varA, varB, false);

        assertFalse(support.isEditAlreadyConsidered(edit1));

        support.markEditAsConsidered(edit1);

        assertTrue(support.isEditAlreadyConsidered(edit1));
        assertFalse(support.isEditAlreadyConsidered(edit2));
    }

    @Test
    public void testReset() {
        BaseLinkEdit edit1 = new AddLinkEdit(probNet, varA, varB, true);

        support.markEditAsConsidered(edit1);
        assertTrue(support.isEditAlreadyConsidered(edit1));

        support.reset();

        assertFalse(support.isEditAlreadyConsidered(edit1));
    }

    @Test
    public void testDuplicateMarkIsIdempotent() {
        BaseLinkEdit edit = new AddLinkEdit(probNet, varA, varB, true);

        support.markEditAsConsidered(edit);
        support.markEditAsConsidered(edit);

        assertTrue(support.isEditAlreadyConsidered(edit));
    }
}
