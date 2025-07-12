/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.pc.independencetester;

import org.openmarkov.core.exception.NodeNotFoundException;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.network.Node;

import java.util.List;

/**
 * This interface represents a general independence tester.
 *
 * @author joliva
 */
public interface IndependenceTester {
    
    /**
     * Tests the dependency level of two variables A and B given an adjacency set of variables Z.
     *
     * @param caseDatabase    <code>caseDatabase</code> case database
     * @param node2           <code>Node</code> second variable.
     * @param adjacencySubset <code>ArrayList</code> of <code>Node</code>
     *                        representing the separation set (i.e. the conditional set).
     * @return the score obtained in the independence test.
     */
    double test(CaseDatabase caseDatabase, Node node1, Node node2, List<Node> adjacencySubset);
}
