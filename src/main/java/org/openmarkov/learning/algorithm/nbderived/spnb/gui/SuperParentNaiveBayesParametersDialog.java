package org.openmarkov.learning.algorithm.nbderived.spnb.gui;

import org.apache.poi.util.StringUtil;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.learning.algorithm.nbderived.spnb.SuperParentNBAlgorithm;
import org.openmarkov.learning.metric.Metric;
import org.openmarkov.learning.metric.annotation.MetricManager;
import org.openmarkov.learning.core.algorithm.LearningAlgorithm;
import org.openmarkov.learning.gui.AlgorithmConfiguration;
import org.openmarkov.learning.gui.AlgorithmParametersDialog;
import org.openmarkov.plugin.service.PluginException;
import static org.openmarkov.learning.algorithm.nbderived.common.util.CommonUtils.LINE_SEPARATOR;
import static org.openmarkov.learning.algorithm.nbderived.common.util.CommonUtils.getStringFromCamelCaseExpression;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


@SuppressWarnings("serial")
@AlgorithmConfiguration(algorithm = SuperParentNBAlgorithm.class)
public class SuperParentNaiveBayesParametersDialog extends AlgorithmParametersDialog {

    private String metric = "Accuracy";
    private static String alphaParameter = "0.5";
    private MetricManager metricManager;

    private JButton AcceptButton;
    private JTextField alphaText;
    private JCheckBox sameSPCheckbox;
    private JLabel jLabel7;
    private JLabel jLabelSP;
    private JPanel jPanel1;

    public SuperParentNaiveBayesParametersDialog(JFrame parent, boolean modal) {
        super(parent, modal);
        setLocationRelativeTo(parent);
        metricManager = new MetricManager();
        initComponents();
    }

    @Override
    public String getDescription() {
        return StringUtil.join(LINE_SEPARATOR,
                Arrays.asList(stringDatabase.getString("Learning.SuperParentNaiveBayes.Metric") + ": "+
                                getStringFromCamelCaseExpression(metric),
                        stringDatabase.getString("Learning.Alpha") + ": " + alphaParameter
                ).toArray());
    }


    @Override
    public LearningAlgorithm getInstance(ProbNet probNet, CaseDatabase database) {
        Metric metricInstance = null;
        try {
            metricInstance = (Metric) Arrays.stream(metricManager.getMetricByName(metric).getConstructors()).iterator().next().newInstance();
        } catch (InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
            e.printStackTrace();
        }
        return new SuperParentNBAlgorithm(probNet, database, metricInstance, 0.0, sameSPCheckbox.isSelected());
    }



    public String getMetric() {
        return metric;
    }



    private void initComponents() {
        jPanel1 = new JPanel();
        alphaText = new JTextField();
        jLabel7 = new JLabel();
        AcceptButton = new JButton();
        jLabelSP = new JLabel();
        sameSPCheckbox = new JCheckBox();
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        setTitle(stringDatabase.getString("Learning.SuperParentNaiveBayes.Title"));
        jPanel1.setBorder(BorderFactory.createTitledBorder(stringDatabase.getString("Learning.SuperParentNaiveBayes.Title")));
        alphaText.setText(alphaParameter);
        jLabel7.setText(stringDatabase.getString("Learning.Alpha") + ":");
        jLabel7.setToolTipText(stringDatabase.getString("Learning.Alpha.Tooltip"));
        jLabelSP.setText(stringDatabase.getString("Learning.SuperParentNaiveBayes.SameSP") + ":");
        jLabelSP.setToolTipText(stringDatabase.getString("Learning.SuperParentNaiveBayes.SameSP.Tooltip"));
        sameSPCheckbox.setSelected(true);

        AcceptButton.setText(stringDatabase.getString("Learning.Ok"));
        AcceptButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                acceptButtonActionPerformed(evt);
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
                                                .addComponent(jLabelSP).addComponent(jLabel7))
                                                       .addPreferredGap(LayoutStyle.ComponentPlacement.UNRELATED).addGroup(
                                                jPanel1Layout
                                                        .createParallelGroup(GroupLayout.Alignment.TRAILING)
                                                        .addComponent(sameSPCheckbox,
                                                                GroupLayout.PREFERRED_SIZE, 40,
                                                                GroupLayout.PREFERRED_SIZE)
                                                        .addComponent(alphaText, GroupLayout.PREFERRED_SIZE,
                                                                40, GroupLayout.PREFERRED_SIZE))))
                        .addContainerGap(12, Short.MAX_VALUE)));
        jPanel1Layout.setVerticalGroup(jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                .addGroup(jPanel1Layout.createSequentialGroup().addGap(18, 18, 18)
                                       .addGroup(jPanel1Layout.createParallelGroup(GroupLayout.Alignment.BASELINE)
                                .addComponent(jLabelSP)
                                .addComponent(sameSPCheckbox, GroupLayout.PREFERRED_SIZE,
                                        GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                                       .addPreferredGap(LayoutStyle.ComponentPlacement.UNRELATED).addGroup(
                                jPanel1Layout.createParallelGroup(GroupLayout.Alignment.BASELINE)
                                        .addComponent(jLabel7)
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





    private void acceptButtonActionPerformed(ActionEvent evt) {
        List<String> errorMessages= new ArrayList<>();

        try {
            double alpha = Double.parseDouble(alphaText.getText());

            if ((alpha < 0) || (alpha > 1)) {
                errorMessages.add(stringDatabase.getString("Learning.SuperParentNaiveBayes.IncorrectParameter"));
            }
        } catch (NumberFormatException e) {
            errorMessages.add(stringDatabase.getString("Learning.SuperParentNaiveBayes.IncorrectParameter"));
        }

        if(!errorMessages.isEmpty()){
            JOptionPane.showMessageDialog(null, StringUtil.join(LINE_SEPARATOR, errorMessages.toArray()),
                    stringDatabase.getString("ErrorWindow.Title.Label"), JOptionPane.ERROR_MESSAGE);
            return;
        }

        alphaParameter = alphaText.getText();
        this.setVisible(false);
    }

}
