/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.em;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.core.exception.*;
import org.openmarkov.core.model.database.CaseDatabase;
import org.openmarkov.core.model.network.EvidenceCase;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.potential.Potential;
import org.openmarkov.core.model.network.potential.PotentialRole;
import org.openmarkov.core.model.network.potential.TablePotential;
import org.openmarkov.core.model.network.potential.UniformPotential;
import org.openmarkov.core.model.network.potential.canonical.ICIPotential;
import org.openmarkov.inference.algorithm.huginPropagation.ClusterPropagation.StorageLevel;
import org.openmarkov.inference.algorithm.huginPropagation.HuginPropagation;
import org.openmarkov.learning.core.algorithm.LearningAlgorithm;
import org.openmarkov.learning.core.algorithm.LearningAlgorithmType;
import org.openmarkov.learning.core.util.LearningEditMotivation;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.learning.core.util.ModelNetUse;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

/**
 * Implements the Expectation-Maximization (EM) parametric learning algorithm
 * for Bayesian networks with latent (unobserved) variables.
 * <p>
 * The EM algorithm iteratively performs two steps:
 * <ul>
 * <li><b>E-step (Expectation):</b> Calculate expected sufficient statistics
 * given current parameters and observed data, using Hugin propagation</li>
 * <li><b>M-step (Maximization):</b> Update parameters to maximize the
 * expected log-likelihood</li>
 * </ul>
 * <p>
 * Convergence: the loop runs until the log-likelihood improvement falls below
 * {@link #EPSILON} or {@link #MAX_ITERATIONS} is reached.
 * <p>
 * Only parametric learning is implemented; structural EM ({@code getBestEdit},
 * {@code getNextEdit}) returns null.
 *
 * @author Iñigo
 * @author Manuel Arias
 * @version 0.3.0-SNAPSHOT
 * @see <a href=
 * "https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm">EM
 * Algorithm on Wikipedia</a>
 * @since OpenMarkov 0.3.0
 */
@LearningAlgorithmType(name = "Expectation maximization (EM)", discriminative = false, supportsUnobservedVariables = true)
public class EMAlgorithm extends LearningAlgorithm {
    
    private static final Logger logger = LogManager.getLogger(EMAlgorithm.class);
    
    private static final double EPSILON = 0.00001;
    private static final int MAX_ITERATIONS = 100;
    
    /**
     * Constructs an EM learning algorithm instance.
     *
     * @param probNet      The probabilistic network to learn parameters for
     * @param caseDatabase The database of cases (with possible missing values)
     * @param alpha        Dirichlet prior strength used by the M-step smoothing
     *                     {@code theta = (count + alpha * prior) / (parentCount + alpha)};
     *                     {@code alpha = 0} gives the maximum-likelihood estimate. The prior
     *                     mean is the network's initial (uniform) CPT.
     */
    public EMAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Double alpha) {
        super(probNet, caseDatabase, alpha);
        // TODO: support a non-uniform prior mean. Currently the prior is the initial
        // (uniform) CPT, so alpha only pulls the estimates toward the uniform distribution.
    }
    
    @Override
    public void init(ModelNetUse modelNetUse) {
        // Nothing here
        
    }
    
    /**
     * Returns the motivation for a given edit.
     * <p>
     * <b>Note:</b> This method is not applicable for EM algorithm as structural
     * learning is not currently implemented. EM only performs parametric learning.
     *
     * @param edit The proposed network edit
     *
     * @return null (structural learning not supported)
     */
    @Override
    public LearningEditMotivation getMotivation(PNEdit edit) {
        // This does not make sense for the time being, as structural learning is not
        // implemented for EM
        return null;
    }
    
    /**
     * Parametric learning
     */
    @Override
    public ProbNet parametricLearning() throws NotEvaluableNetworkException.NotApplicableNetwork, ConstraintViolatedException, IncompatibleEvidenceException.EvidenceIsIncompatibleWithOther, NonProjectablePotentialException, CannotNormalizePotentialException {
        int[][] cases = caseDatabase.getCases();
        List<Variable> variables = caseDatabase.getVariables();

        if (cases.length == 0) {
            // No data: there is nothing to learn, so leave the network untouched.
            return probNet;
        }

        // Init sigma
        List<TablePotential> potentials = new ArrayList<>();
        Map<ICIPotential, List<TablePotential>> iciSubpotentials = new HashMap<>();
        ProbNet expandedNet = adaptNetwork(probNet, potentials, iciSubpotentials);
        
        HashMap<Potential, TablePotential> expertKnowledge = new HashMap<Potential, TablePotential>();
        for (TablePotential potential : potentials) {
            expertKnowledge.put(potential, new TablePotential(potential));
        }
        
        double lastLogLikelihood = Double.NEGATIVE_INFINITY;
        double currentLogLikelihood = Double.NEGATIVE_INFINITY;
        boolean converged = false;
        int iterations = 0;
        
        do {
            HashMap<Potential, TablePotential> expectedCountsMap = new HashMap<>();

            // E-step: compute expected sufficient statistics for each case
            for (int i = 0; i < cases.length; ++i) {
                // A fresh inference engine per case. ClusterPropagation (StorageLevel.FULL)
                // caches cluster messages and posteriors and does NOT invalidate them when
                // new post-resolution evidence is set, so reusing one instance across cases
                // would return the first case's joint for every case. Recreating it per case
                // also picks up the M-step changes to the potentials from the previous
                // iteration (ClusterPropagation copies the network internally).
                HuginPropagation inferenceAlgorithm = new HuginPropagation(expandedNet);
                inferenceAlgorithm.setStorageLevel(StorageLevel.FULL);
                // Compute the joint of every conditional potential once for this case (keyed by
                // its conditioned variable). Calling this once per potential instead would repeat
                // the whole inference O(P) times per potential, i.e. O(P^2) per case.
                Map<Variable, TablePotential> caseJointProbabilities =
                        new JointProbabilityCalculator(variables, cases[i], inferenceAlgorithm, expandedNet).call();
                for (Potential potential : potentials) {
                    TablePotential jointProbability = caseJointProbabilities.get(potential.getVariable(0));
                    if (expectedCountsMap.containsKey(potential)) {
                        sum(expectedCountsMap.get(potential), jointProbability);
                    } else {
                        expectedCountsMap.put(potential, jointProbability);
                    }
                }
            }
            
            // M-step: update parameters to maximize expected log-likelihood
            for (TablePotential potential : potentials) {
                Variable childVariable = potential.getVariables().get(0);
                int childNumStates = childVariable.getNumStates();
                double[] theta = potential.getValues();
                double[] p_ijk = expertKnowledge.get(potential).getValues();
                double[] expectedCounts = expectedCountsMap.get(potential).getValues();
                double[] expectedCountsParents = new double[expectedCounts.length / childNumStates];
                
                // Marginalize child variable: M[x,u] -> M[u]
                for (int i = 0; i < expectedCounts.length; ++i) {
                    expectedCountsParents[i / childNumStates] += expectedCounts[i];
                }
                
                // Calculate new theta (Madsen 2003)
                for (int i = 0; i < theta.length; ++i) {
                    double denominator = expectedCountsParents[i / childNumStates] + alpha;
                    // A parent configuration never seen in the data (and with no Dirichlet prior,
                    // alpha=0) has denominator 0. There is nothing to estimate for it, so keep its
                    // current parameters: dividing 0/0 would write NaN and the next iteration's
                    // inference would then propagate NaN across the whole CPT.
                    if (denominator != 0.0) {
                        theta[i] = (expectedCounts[i] + alpha * p_ijk[i]) / denominator;
                    }
                }
            }
            
            // Convergence measure: the expected complete-data log-likelihood (the EM
            // Q-function) evaluated at the updated parameters, sum over cells of
            // expectedCounts * log(theta). This is the M-step objective, not the observed-data
            // log-likelihood; it is used only to detect when successive iterations stop
            // improving. Cells with zero expected count contribute 0 and are skipped (also
            // avoiding log(0) for parameters legitimately driven to 0 by MLE).
            currentLogLikelihood = 0.0;
            for (TablePotential potential : potentials) {
                TablePotential expectedCounts = expectedCountsMap.get(potential);
                double[] theta = potential.getValues();
                for (int i = 0; i < theta.length; ++i) {
                    if (expectedCounts.getValues()[i] > 0) {
                        currentLogLikelihood += expectedCounts.getValues()[i] * Math.log(theta[i]);
                    }
                }
            }

            // Stop once the improvement drops below EPSILON. The quantity increases toward
            // convergence, so a non-positive change (<= EPSILON) also stops the loop.
            converged = (currentLogLikelihood - lastLogLikelihood) <= EPSILON;
            lastLogLikelihood = currentLogLikelihood;
            ++iterations;
            logger.debug("EM iteration {}: log-likelihood = {}", iterations, currentLogLikelihood);
            
        } while (!converged && iterations < MAX_ITERATIONS);
        
        logger.info("EM finished after {} iterations (converged={}, logLikelihood={})",
                    iterations, converged, currentLogLikelihood);
        
        for (ICIPotential iciPotential : iciSubpotentials.keySet()) {
            iciPotential.setNoisyPotentials(iciSubpotentials.get(iciPotential));
        }
        
        return probNet;
    }
    
    private static void sum(TablePotential tablePotential, TablePotential jointProbability) {
        double[] tablePotentialValues = tablePotential.getValues();
        double[] jointProbabilityValues = jointProbability.getValues();
        
        for (int i = 0; i < tablePotentialValues.length; ++i) {
            tablePotentialValues[i] += jointProbabilityValues[i];
        }
    }
    
    private static ProbNet adaptNetwork(ProbNet probNet, List<TablePotential> potentials,
                                        Map<ICIPotential, List<TablePotential>> iciSubpotentials) {
        ProbNet expandedNet = probNet.copy();
        for (Potential potential : expandedNet.getPotentials()) {
            if (potential.getPotentialRole() == PotentialRole.CONDITIONAL_PROBABILITY) {
                switch (potential) {
                    case UniformPotential uniformPotential -> {
                        // A UniformPotential is not a TablePotential (sibling types), so it cannot be
                        // copied by casting. Build an explicit table from its variables/role: the
                        // TablePotential constructor initialises it to a uniform distribution
                        // (setUniform()), which is exactly the EM starting point we want.
                        TablePotential newPotential = new TablePotential(uniformPotential.getVariables(),
                                                                         uniformPotential.getPotentialRole());
                        potentials.add(newPotential);
                        expandedNet.getNode(potential.getVariable(0)).setPotential(newPotential);
                    }
                    case ICIPotential iciPotential -> {
                        List<TablePotential> noisyPotentials = iciPotential.getNoisyPotentials();
                        iciSubpotentials.put(iciPotential, noisyPotentials);
                        potentials.addAll(noisyPotentials);
                        
                        Variable conditioningVariable = potential.getVariable(0);
                        for (TablePotential noisyPotential : noisyPotentials) {
                            Variable zVariable = noisyPotential.getVariable(0);
                            Variable parentVariable = noisyPotential.getVariable(1);
                            expandedNet.addNode(zVariable, NodeType.CHANCE);
                            expandedNet.removeLink(parentVariable, conditioningVariable, true);
                            expandedNet.addLink(parentVariable, zVariable, true);
                            expandedNet.addLink(zVariable, conditioningVariable, true);
                            expandedNet.getNode(zVariable).setPotential(noisyPotential);
                        }
                        TablePotential leakyPotential = iciPotential.getLeakyPotential();
                        if (leakyPotential != null) {
                            Variable leakyVariable = leakyPotential.getVariable(0);
                            expandedNet.addNode(leakyVariable, NodeType.CHANCE);
                            expandedNet.getNode(leakyVariable).setPotential(leakyPotential);
                            expandedNet.addLink(leakyVariable, conditioningVariable, true);
                        }
                        expandedNet.getNode(conditioningVariable).setPotential(iciPotential.getFFunctionPotential());
                    }
                    case TablePotential tablePotential -> potentials.add(tablePotential);
                    default -> {
                    }
                }
            }
        }
        return expandedNet;
    }
    
    /**
     * Returns the best structural edit proposal.
     * <p>
     * <b>Not Implemented:</b> EM algorithm currently only supports parametric
     * learning.
     * Structural EM would require implementing this method to propose structure
     * changes.
     *
     * @param onlyAllowedEdits  If true, only return edits that satisfy constraints
     * @param onlyPositiveEdits If true, only return edits with positive score
     *
     * @return null (structural learning not implemented)
     */
    @Override
    public LearningEditProposal getBestEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        // TODO: Implement structural EM if needed
        return null;
    }
    
    /**
     * Returns the next best structural edit proposal.
     * <p>
     * <b>Not Implemented:</b> EM algorithm currently only supports parametric
     * learning.
     *
     * @param onlyAllowedEdits  If true, only return edits that satisfy constraints
     * @param onlyPositiveEdits If true, only return edits with positive score
     *
     * @return null (structural learning not implemented)
     */
    @Override
    public LearningEditProposal getNextEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        // TODO: Implement structural EM if needed
        return null;
    }
    
    private static class JointProbabilityCalculator implements Callable<Map<Variable, TablePotential>> {
        private List<Variable> variables;
        private int[] dataCase;
        private HuginPropagation inferenceAlgorithm;
        private ProbNet expandedNet;
        
        /**
         * Constructor for JointProbabilityCalculator.
         *
         * @param variables          the variables
         * @param dataCase           the data case
         * @param inferenceAlgorithm the inference algorithm
         * @param expandedNet        the expanded net
         */
        public JointProbabilityCalculator(List<Variable> variables, int[] dataCase, HuginPropagation inferenceAlgorithm,
                                          ProbNet expandedNet) {
            this.variables = variables;
            this.dataCase = dataCase;
            this.inferenceAlgorithm = inferenceAlgorithm;
            this.expandedNet = expandedNet;
        }
        
        @Override
        public Map<Variable, TablePotential> call() throws IncompatibleEvidenceException.EvidenceIsIncompatibleWithOther, NonProjectablePotentialException, CannotNormalizePotentialException {
            Map<Variable, TablePotential> jointProbabilities = new HashMap<>();
            EvidenceCase caseEvidence = new EvidenceCase();
            for (int j = 0; j < dataCase.length; ++j) {
                Variable variable = variables.get(j);
                String stateName = variable.getStateName(dataCase[j]);
                if (!stateName.equals("?")) {
                    caseEvidence.addFinding(expandedNet, variable.getName(), stateName);
                }
            }
            inferenceAlgorithm.setPostResolutionEvidence(caseEvidence);
            for (Potential potential : expandedNet.getPotentials()) {
                if (potential.getPotentialRole() == PotentialRole.CONDITIONAL_PROBABILITY) {
                    Variable conditioningVariable = potential.getVariable(0);
                    // getJointProbability returns the joint in the junction-tree cluster's variable
                    // order, which need not match the CPT's [child, parents] order. Realign it so the
                    // M-step (which marginalises the child assuming it is the fastest-varying index)
                    // sums and updates the right cells.
                    TablePotential jointProbability = inferenceAlgorithm.getJointProbability(potential.getVariables())
                                                                        .reorder(potential.getVariables());
                    jointProbabilities.put(conditioningVariable, jointProbability);
                }
            }
            return jointProbabilities;
        }
        
    }
    
}
