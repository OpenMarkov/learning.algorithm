/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */
package org.openmarkov.learning.algorithm.pc.util;

import org.openmarkov.core.model.network.Node;
import org.openmarkov.learning.core.util.LearningEditMotivation;
import org.openmarkov.learning.core.util.ScoreEditMotivation;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.List;
import java.util.Locale;

public class PCEditMotivation extends ScoreEditMotivation {

	protected List<Node> separationSet;

	public PCEditMotivation(double score, List<Node> separationSet) {
		super(score);
		this.separationSet = separationSet;
	}

	public List<Node> getSeparationSet() {
		return separationSet;
	}

	public String toString() {

		String description = "{";
		for (Node node : separationSet) {
			description += node.getName() + ", ";
		}

		DecimalFormat df = new DecimalFormat("0.0000", new DecimalFormatSymbols(Locale.US));

		if (description.length() > 1)
			description = description.substring(0, description.length() - 2) + "} p: " + df.format(getScore());
		else
			description += "} p: " + df.format(getScore());

		return description;
	}

	public int compareTo(LearningEditMotivation otherEdit) {
		int returnValue = 0;

		if (otherEdit == null) {
			returnValue = 1;
		} else if (otherEdit instanceof PCEditMotivation) {
			int otherSeparationSetSize = ((PCEditMotivation) otherEdit).getSeparationSet().size();
			if (otherSeparationSetSize > separationSet.size()) {
				returnValue = 1;
			} else if (otherSeparationSetSize < separationSet.size()) {
				returnValue = -1;
			} else {
				returnValue = super.compareTo(otherEdit);
			}
		}
		return returnValue;
	}
}
