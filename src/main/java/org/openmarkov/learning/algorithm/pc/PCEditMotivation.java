/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */
package org.openmarkov.learning.algorithm.pc;

import org.openmarkov.core.model.network.Node;
import org.openmarkov.learning.core.util.LearningEditMotivation;
import org.openmarkov.learning.core.util.ScoreEditMotivation;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.List;
import java.util.Locale;

/**
 * {@code PCEditMotivation} stores the justification (motivation) for an edit
 * in the PC algorithm. It contains both the statistical score (e.g., p-value)
 * obtained from an independence test and the separation set that led to this decision.
 * <p>
 * It is mainly used to record why a link was oriented or removed during
 * the causal discovery process.
 */
public class PCEditMotivation extends ScoreEditMotivation {

    /**
     * This is the set of variables conditioned upon in the independence test.
     */
	protected List<Node> separationSet;

   /**
     * Creates a new {@code PCEditMotivation}.
     *
     * @param score         The statistical score associated with the test
     *                      (usually a p-value).
     * @param separationSet The set of nodes that forms the separation set
     *                      between the two tested variables.
     */
	public PCEditMotivation(double score, List<Node> separationSet) {
		super(score);
		this.separationSet = separationSet;
	}

	public List<Node> getSeparationSet() {
		return separationSet;
	}

    /**
     * Returns a textual description of this motivation, including the separation set
     * and the score.
     *
     * @return A string in the format "{X, Y, Z} p: value".
     */
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

    /**
     * Compares this motivation with another {@code LearningEditMotivation}.
     * <p>
     * The comparison is based on:
     * <ol>
     *     <li>The size of the separation set (larger separation sets are considered greater).</li>
     *     <li>If the sizes are equal, the comparison defined in {@link ScoreEditMotivation} is used.</li>
     * </ol>
     *
     * @param otherEdit another edit motivation.
     * @return A negative integer, zero, or a positive integer as this motivation
     *         is less than, equal to, or greater than the specified one.
     */
	@Override public int compareTo(LearningEditMotivation otherEdit) {
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
