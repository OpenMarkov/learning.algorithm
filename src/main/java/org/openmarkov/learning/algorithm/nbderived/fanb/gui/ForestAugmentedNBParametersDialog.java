package org.openmarkov.learning.algorithm.nbderived.fanb.gui;

import org.apache.poi.util.StringUtil;
import org.jetbrains.annotations.Nullable;
import org.openmarkov.core.exception.InvalidArgumentException;
import org.openmarkov.core.exception.UnrecoverableException;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.learning.algorithm.nbderived.fanb.ForestAugmentedNBAlgorithm;
import org.openmarkov.learning.metric.Metric;
import org.openmarkov.learning.metric.annotation.MetricManager;
import org.openmarkov.learning.core.algorithm.LearningAlgorithm;
import org.openmarkov.learning.gui.AlgorithmConfiguration;
import org.openmarkov.learning.gui.AlgorithmParametersDialog;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.openmarkov.learning.algorithm.nbderived.common.util.CommonUtils.LINE_SEPARATOR;
import static org.openmarkov.learning.algorithm.nbderived.common.util.CommonUtils.getStringFromCamelCaseExpression;

@SuppressWarnings("serial")
@AlgorithmConfiguration(algorithm = ForestAugmentedNBAlgorithm.class)
public class ForestAugmentedNBParametersDialog extends AlgorithmParametersDialog {

    private String unconditionedMetric = "MutualInformation";
    private String conditionedMetric = "ConditionalMutualInformation";
    private static String alphaParameter = "0.5";
    private MetricManager metricManager;

    private JButton AcceptButton;
    private JTextField alphaText;
    private JLabel alphaLabel;
    private JPanel jPanel1;



    public ForestAugmentedNBParametersDialog(JFrame parent, boolean modal) {
        super(parent, modal);
        setLocationRelativeTo(parent);
        metricManager = new MetricManager();
        initComponents();
    }


    @Override
    public LearningAlgorithm getInstance(ProbNet probNet, CaseDatabase database) {
        return new ForestAugmentedNBAlgorithm(probNet, database,
                getMetricByName(conditionedMetric),
                getMetricByName(unconditionedMetric),
                Double.parseDouble(alphaParameter));
    }

    private Metric getMetricByName(String metricName){
        Metric metric = null;
        try{
            metric = (Metric) Arrays.stream(metricManager.getMetricByName(metricName).getConstructors()).iterator().next().newInstance();
        } catch (InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
            e.printStackTrace();
        }
        return metric;
    }

    @Override
    public String getDescription() {
        return StringUtil.join(LINE_SEPARATOR,
                Arrays.asList(stringDatabase.getString("Learning.FANB.Metrics") + ": "+
                                getStringFromCamelCaseExpression(unconditionedMetric)+", "+
                                getStringFromCamelCaseExpression(conditionedMetric),
                        stringDatabase.getString("Learning.Alpha") + ": " + alphaParameter
                ).toArray());
    }
    @Override
    public ArrayList<Object> getOptions(){
        ArrayList<Object> options=new ArrayList<>();
        options.add(getMetricByName(conditionedMetric));
        options.add(getMetricByName(unconditionedMetric));
        options.add(Double.parseDouble(alphaParameter));
        return options;
    }

    public String getMetric() {
        return unconditionedMetric;
    }




    private void initComponents() {
        jPanel1 = new JPanel();
        alphaText = new JTextField();
        alphaLabel = new JLabel();
        AcceptButton = new JButton();
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        setTitle(stringDatabase.getString("Learning.FANB.Title"));
        jPanel1.setBorder(BorderFactory.createTitledBorder(stringDatabase.getString("Learning.FANB.Title")));
        alphaText.setText(alphaParameter);
        alphaLabel.setText(stringDatabase.getString("Learning.Alpha") + ":");
        alphaLabel.setToolTipText(stringDatabase.getString("Learning.Alpha.Tooltip"));


        AcceptButton.setText(stringDatabase.getString("Learning.Ok"));
        AcceptButton.addActionListener(evt -> {
            try {
                acceptButtonActionPerformed(evt);
            } catch (InvalidArgumentException e) {
                throw new UnrecoverableException(e);
            }
        });

        GroupLayout jPanel1Layout = new GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                .addGroup(jPanel1Layout.createSequentialGroup().addGroup(
                        jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                                .addGroup(
                                        jPanel1Layout.createSequentialGroup().addGap(92, 92, 92).addComponent(AcceptButton))
                                .addGroup(jPanel1Layout.createSequentialGroup().addContainerGap().addGroup(
                                        jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                                                .addComponent(alphaLabel))
                                                       .addPreferredGap(LayoutStyle.ComponentPlacement.UNRELATED).addGroup(
                                                jPanel1Layout
                                                        .createParallelGroup(GroupLayout.Alignment.TRAILING)
                                                        .addComponent(alphaText, GroupLayout.PREFERRED_SIZE,
                                                                40, GroupLayout.PREFERRED_SIZE))))
                        .addContainerGap(12, Short.MAX_VALUE)));
        jPanel1Layout.setVerticalGroup(jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                .addGroup(jPanel1Layout.createSequentialGroup().addGap(18, 18, 18)
                                       .addPreferredGap(LayoutStyle.ComponentPlacement.UNRELATED).addGroup(
                                jPanel1Layout.createParallelGroup(GroupLayout.Alignment.BASELINE)
                                        .addComponent(alphaLabel)
                                        .addComponent(alphaText, GroupLayout.PREFERRED_SIZE,
                                                GroupLayout.DEFAULT_SIZE,
                                                GroupLayout.PREFERRED_SIZE)).addGap(11, 11, 11)
                                       .addComponent(AcceptButton).addContainerGap()));
        GroupLayout layout = new GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING).addGroup(
                layout.createSequentialGroup().addContainerGap()
                        .addComponent(jPanel1, GroupLayout.PREFERRED_SIZE,
                                GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
                        .addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)));
        layout.setVerticalGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING).addGroup(
                layout.createSequentialGroup().addContainerGap()
                        .addComponent(jPanel1, GroupLayout.PREFERRED_SIZE,
                                GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
                        .addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)));
        pack();
    }
    
    
    private void acceptButtonActionPerformed(ActionEvent evt) throws InvalidArgumentException {
        @Nullable Double alpha;
        try {
            alpha = Double.parseDouble(alphaText.getText());
        } catch (NumberFormatException e) {
            alpha = null;
        }
        if (alpha == null || (alpha < 0) || (alpha > 1)) {
            throw new InvalidArgumentException(alpha, "alpha", "must between 0 and 1");
        }
        alphaParameter = alphaText.getText();
        this.setVisible(false);
    }

}
