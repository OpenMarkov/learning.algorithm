/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */

package org.openmarkov.learning.algorithm.em;

import org.openmarkov.core.action.PNEdit;
import org.openmarkov.core.exception.IncompatibleEvidenceException;
import org.openmarkov.core.exception.NotEvaluableNetworkException;
import org.openmarkov.core.io.database.CaseDatabase;
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
 * Implements Expectation Maximization parametric learning algorithm
 * Maybe Structural EM will be implemented in the future
 *
 * @author Iñigo
 */
//TODO: This learning algorith is disabled
@LearningAlgorithmType(name = "Expectation maximization (EM)", discriminative = false, supportsUnobservedVariables = true)
public class EMAlgorithm extends LearningAlgorithm {
    
    private static final double EPSILON = 0.00001;
    
    public EMAlgorithm(ProbNet probNet, CaseDatabase caseDatabase, Double alpha) {
        super(probNet, caseDatabase, alpha);
        // TODO do something with alpha parameter, e.g. initialize non-latent variables
    }
    
    @Override public void init(ModelNetUse modelNetUse) {
        // Nothing here
        
    }
    
    @Override public LearningEditMotivation getMotivation(PNEdit edit) {
        // This does not make sense for the time being, as structural learning is not implemented for EM
        return null;
    }
    
    /**
     * Parametric learning
     */
    @Override public ProbNet parametricLearning() {
        int[][] cases = caseDatabase.getCases();
        List<Variable> variables = caseDatabase.getVariables();
        
        //Init sigma
        List<TablePotential> potentials = new ArrayList<>();
        Map<ICIPotential, List<TablePotential>> iciSubpotentials = new HashMap<>();
        ProbNet expandedNet = adaptNetwork(probNet, potentials, iciSubpotentials);
        
        HashMap<Potential, TablePotential> expertKnowledge = new HashMap<Potential, TablePotential>();
        
        for (TablePotential potential : potentials) {
            expertKnowledge.put(potential, new TablePotential(potential));
        }
        
        HuginPropagation inferenceAlgorithm = null;
        try {
            inferenceAlgorithm = new HuginPropagation(expandedNet);
            inferenceAlgorithm.setStorageLevel(StorageLevel.FULL);
        } catch (NotEvaluableNetworkException.NotApplicableNetwork|NotEvaluableNetworkException.UnsatisfiedContraints e1) {
            e1.printStackTrace();
        }
        
        double lastLogLikelihood = Double.NEGATIVE_INFINITY;
        double currentLogLikelihood = Double.NEGATIVE_INFINITY;
        
        int iterations = 0;
        do {
            HashMap<Potential, TablePotential> expectedCountsMap = new HashMap<Potential, TablePotential>();
            
            //E-step
            //For each case in the database
            int notNull = 0;
            // List<Double> accruedWeights = new ArrayList<> (cases.length);
            for (int i = 0; i < cases.length; ++i) {
                Map<Variable, TablePotential> jointProbabilities =
                        new JointProbabilityCalculator(variables, cases[i], inferenceAlgorithm, expandedNet).call();
                if (jointProbabilities != null) {
                    notNull++;
                    System.out.println(notNull + " from " + i);
                    for (Potential potential : potentials) {
                        TablePotential jointProbability = jointProbabilities.get(potential.getVariable(0));
                        if (expectedCountsMap.containsKey(potential)) {
                            sum(expectedCountsMap.get(potential), jointProbability);
                        } else {
                            expectedCountsMap.put(potential, jointProbability);
                        }
                    }
                }
                // accruedWeights.add (inferenceAlgorithm.getAccruedWeight ());
            }
            // System.out.println(accruedWeights.toString ());
            //M-step
            for (TablePotential potential : potentials) {
                Variable childVariable = potential.getVariables().get(0);
                int childNumStates = childVariable.getNumStates();
                double[] theta = potential.values;
                double[] p_ijk = expertKnowledge.get(potential).values;
                double[] expectedCounts = expectedCountsMap.get(potential).values;
                double[] expectedCountsParents = new double[expectedCounts.length / childNumStates];
                
                // Marginalize child variable: M[x,u]-> M[u]
                for (int i = 0; i < expectedCounts.length; ++i) {
                    expectedCountsParents[i / childNumStates] += expectedCounts[i];
                }
                
                // Calculate new theta (as seen on madsen2003)
                for (int i = 0; i < theta.length; ++i) {
                    theta[i] = (expectedCounts[i] + alpha * p_ijk[i]) / (
                            expectedCountsParents[i / childNumStates] + alpha
                    );
                }
            }
            
            // Calculate new log likelihood
            lastLogLikelihood = currentLogLikelihood;
            currentLogLikelihood = 0.0;
            for (TablePotential potential : potentials) {
                TablePotential expectedCounts = expectedCountsMap.get(potential);
                double[] theta = potential.values;
                for (int i = 0; i < theta.length; ++i) {
                    if (expectedCounts.values[i] > 0) {
                        currentLogLikelihood += expectedCounts.values[i] * Math.log(theta[i]);
                    }
                }
                
            }
            ++iterations;
            
        } while (false);//iterations  < 100 && (currentLogLikelihood - lastLogLikelihood) > EPSILON);
        
        for (ICIPotential iciPotential : iciSubpotentials.keySet()) {
            iciPotential.setNoisyPotentials(iciSubpotentials.get(iciPotential));
        }
        
        return probNet;
    }
    
    private void sum(TablePotential tablePotential, TablePotential jointProbability) {
        double[] tablePotentialValues = tablePotential.values;
        double[] jointProbabilityValues = jointProbability.values;
        
        for (int i = 0; i < tablePotentialValues.length; ++i) {
            tablePotentialValues[i] += jointProbabilityValues[i];
        }
    }
    
    private ProbNet adaptNetwork(ProbNet probNet, List<TablePotential> potentials,
                                 Map<ICIPotential, List<TablePotential>> iciSubpotentials) {
        ProbNet expandedNet = probNet.copy();
        for (Potential potential : expandedNet.getPotentials()) {
            if (potential.getPotentialRole() == PotentialRole.CONDITIONAL_PROBABILITY) {
                if (potential instanceof UniformPotential) {
                    TablePotential newPotential = new TablePotential((TablePotential) potential);
                    potentials.add(newPotential);
                    expandedNet.getNode(potential.getVariable(0)).setPotential(newPotential);
                } else if (potential instanceof ICIPotential) {
                    ICIPotential iciPotential = (ICIPotential) potential;
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
                    
                } else if (potential instanceof TablePotential) {
                    potentials.add((TablePotential) potential);
                }
            }
        }
        return expandedNet;
    }
    
    @Override public LearningEditProposal getBestEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        // TODO Auto-generated method stub
        return null;
    }
    
    @Override public LearningEditProposal getNextEdit(boolean onlyAllowedEdits, boolean onlyPositiveEdits) {
        // TODO Auto-generated method stub
        return null;
    }
    
    private class JointProbabilityCalculator implements Callable<Map<Variable, TablePotential>> {
        private List<Variable> variables;
        private int[] dataCase;
        private HuginPropagation inferenceAlgorithm;
        private ProbNet expandedNet;
        
        /**
         * Constructor for JointProbabilityCalculator.
         *
         * @param variables
         * @param dataCase
         * @param inferenceAlgorithm
         * @param expandedNet
         */
        public JointProbabilityCalculator(List<Variable> variables, int[] dataCase, HuginPropagation inferenceAlgorithm,
                                          ProbNet expandedNet) {
            this.variables = variables;
            this.dataCase = dataCase;
            this.inferenceAlgorithm = inferenceAlgorithm;
            this.expandedNet = expandedNet;
        }
        
        @Override public Map<Variable, TablePotential> call() {
            Map<Variable, TablePotential> jointProbabilities = new HashMap<>();
            EvidenceCase caseEvidence = new EvidenceCase();
            for (int j = 0; j < dataCase.length; ++j) {
                Variable variable = variables.get(j);
                try {
                    String stateName = variable.getStateName(dataCase[j]);
                    if (!stateName.equals("?")) {
                        caseEvidence.addFinding(expandedNet, variable.getName(), stateName);
                    }
                } catch (IncompatibleEvidenceException.EvidenceIsIncompatibleWithOther e) {
                    e.printStackTrace();
                }
            }
            inferenceAlgorithm.setPostResolutionEvidence(caseEvidence);
            for (Potential potential : expandedNet.getPotentials()) {
                if (potential.getPotentialRole() == PotentialRole.CONDITIONAL_PROBABILITY) {
                    Variable conditioningVariable = potential.getVariable(0);
                    jointProbabilities.put(conditioningVariable,
                                           inferenceAlgorithm.getJointProbability(potential.getVariables()));
                }
            }
            return jointProbabilities;
        }
        
    }
    
}
