package org.openmarkov.learning.algorithm.naivebayes.gui;

import org.apache.poi.util.StringUtil;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.learning.algorithm.naivebayes.NaiveBayesAlgorithm;
import org.openmarkov.learning.metric.annotation.MetricManager;
import org.openmarkov.learning.core.algorithm.LearningAlgorithm;
import org.openmarkov.learning.gui.AlgorithmConfiguration;
import org.openmarkov.learning.gui.AlgorithmParametersDialog;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@SuppressWarnings("serial")
@AlgorithmConfiguration(algorithm = NaiveBayesAlgorithm.class)
public class NaiveBayesParametersDialog extends AlgorithmParametersDialog {

    private static String LINE_SEPARATOR = System.getProperty("line.separator");
    
    private String unconditionedMetric = "MutualInformation";
    private static String alphaParameter = "0.5";
    private MetricManager metricManager;

    private JButton AcceptButton;
    private JTextField alphaText;
    private JLabel alphaLabel;
    private JPanel jPanel1;

    public NaiveBayesParametersDialog(JFrame parent, boolean modal) {
        super(parent, modal);
        setLocationRelativeTo(parent);
        initComponents();
    }

    @Override
    public String getDescription() {
        return StringUtil.join(LINE_SEPARATOR,
                Arrays.asList(
                        stringDatabase.getString("Learning.Alpha") + ": " + alphaParameter
                ).toArray());
    }



    @Override
    public LearningAlgorithm getInstance(ProbNet probNet, CaseDatabase database) {
        return new NaiveBayesAlgorithm(probNet, database, 0.0);
    }



    private void initComponents() {
        jPanel1 = new JPanel();
        alphaText = new JTextField();
        alphaLabel = new JLabel();
        AcceptButton = new JButton();
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        setTitle(stringDatabase.getString("Learning.NaiveBayes.Title"));
        jPanel1.setBorder(BorderFactory.createTitledBorder(stringDatabase.getString("Learning.NaiveBayes.Title")));
        alphaText.setText(alphaParameter);
        alphaLabel.setText(stringDatabase.getString("Learning.Alpha") + ":");
        alphaLabel.setToolTipText(stringDatabase.getString("Learning.Alpha.Tooltip"));


        AcceptButton.setText(stringDatabase.getString("Learning.Ok"));
        AcceptButton.addActionListener(new java.awt.event.ActionListener() {
            @Override public void actionPerformed(ActionEvent evt) {
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


    private void acceptButtonActionPerformed(ActionEvent evt) {
        List<String> errorMessages= new ArrayList<>();

        try {
            double alpha = Double.parseDouble(alphaText.getText());

            if ((alpha < 0) || (alpha > 1)) {
                errorMessages.add(stringDatabase.getString("Learning.NaiveBayes.IncorrectParameter"));
            }
        } catch (NumberFormatException e) {
            errorMessages.add(stringDatabase.getString("Learning.NaiveBayes.IncorrectParameter"));
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
