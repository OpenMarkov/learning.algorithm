/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.pc.independencetester;

import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;

/**
 * Interface for testing the causal direction of a link between two nodes.
 *
 * <p>Implementations assign a score to the hypothesis X → Y. The PC algorithm
 * uses this in the {@code REMAINING_LINKS_ORIENTATION} phase to orient links
 * that cannot be determined by acyclicity and v-structure rules alone.
 *
 * <p>The score is typically a p-value: a high value means the data support the
 * direction X → Y under the chosen causal model.
 */
public interface CausalDirectionTester {

    /**
     * Returns a score for the hypothesis that {@code nodeX} causally precedes
     * {@code nodeY}.
     *
     * @param caseDatabase the case database
     * @param nodeX        the candidate cause node
     * @param nodeY        the candidate effect node
     * @return a score where higher values support the X → Y direction
     */
    double testDirection(CaseDatabase caseDatabase, Node nodeX, Node nodeY);
}
