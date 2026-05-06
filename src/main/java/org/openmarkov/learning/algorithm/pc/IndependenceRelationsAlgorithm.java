/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.pc;

import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.graph.Link;
import org.openmarkov.core.model.network.Node;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.learning.core.algorithm.LearningAlgorithm;
import org.openmarkov.learning.core.util.ModelNetUse;

/**
 * This class implements the basic structure of any algorithm based on the
 * independence relations approach.
 * The particular behavior of each algorithm is given by the
 * <i>editionsGenerator</i> class.
 *
 * @author joliva
 * @author manuel
 * @author fjdiez
 * @version 1.0
 * @since OpenMarkov 1.0
 */
public abstract class IndependenceRelationsAlgorithm extends LearningAlgorithm {

	/**  Intended for future use. */
	protected boolean undirectedStructureFound = false;

	/**
	 * Constructs an independence-relations-based learning algorithm.
	 *
	 * @param probNet      the probabilistic network to learn
	 * @param caseDatabase the case database to learn from
	 * @param alpha        the significance level for independence tests
	 */
	public IndependenceRelationsAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, double alpha) {
		super(probNet, caseDatabase, alpha);
	}

	/**
	 * Initializes the algorithm depending on the model net use selected
	 * by the user.
	 *
	 * @param modelNetUse the model network usage configuration
	 */
	@Override public void init(ModelNetUse modelNetUse) {

		/* If the user allowed links addition, the algorithm starts
		 * with the complete graph
		 */
		if (modelNetUse != null && (modelNetUse.isLinkAdditionAllowed() || modelNetUse.isUseNodePositions())) {
			for (Link<Node> link : probNet.getLinks()) {
				probNet.removeLink(link);
			}
			probNet.marry(probNet.getNodes());
		}
		/* If the user allowed only links deletion, the algorithm starts
		 * with the structure of the model net.
		 */
		else if (modelNetUse != null && (modelNetUse.isLinkRemovalAllowed() || modelNetUse.isLinkInversionAllowed())) {
			for (Link<Node> link : probNet.getLinks()) {
				probNet.removeLink(link);
                probNet.addLink(link.getFrom(), link.getTo(), false);
			}
		} else {
			for (Link<Node> link : probNet.getLinks()) {
				probNet.removeLink(link);
			}
			probNet.marry(probNet.getNodes());
		}

	}

}
