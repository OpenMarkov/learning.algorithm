package org.openmarkov.learning.algorithm.nbderived.kdb.gui;

import org.apache.poi.util.StringUtil;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.learning.algorithm.nbderived.kdb.KDBAlgorithm;
import org.openmarkov.learning.algorithm.nbderived.common.util.CommonUtils;
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

@SuppressWarnings("serial")
@AlgorithmConfiguration(algorithm = KDBAlgorithm.class)
public class KDBParametersDialog extends AlgorithmParametersDialog {

    private String unconditionedMetric = "MutualInformation";
    private String conditionedMetric = "ConditionalMutualInformation";
    private int kValue = 0;
    private static String alphaParameter = "0.5";
    private MetricManager metricManager;

    private JButton AcceptButton;
    private JLabel kValueLabel;
    private JTextField alphaText;
    private JTextField kValueField;
    private JLabel jLabel7;
    private JPanel jPanel1;
    private JComboBox<String> kValueComboBox;

    /**
     * Creates new form PCOptionsGUI
     */
    public KDBParametersDialog(JFrame parent, boolean modal) {
        super(parent, modal);
        setLocationRelativeTo(parent);
        metricManager = new MetricManager();
        initComponents();
    }


    private void initComponents() {
        jPanel1 = new JPanel();
        alphaText = new JTextField();
        jLabel7 = new JLabel();
        AcceptButton = new JButton();
        kValueLabel = new JLabel();
        kValueField = new JTextField();
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        setTitle(stringDatabase.getString("Learning.KDB.Title"));
        jPanel1.setBorder(BorderFactory.createTitledBorder(stringDatabase.getString("Learning.KDB.Title")));
        alphaText.setText(alphaParameter);
        jLabel7.setText(stringDatabase.getString("Learning.Alpha") + ":");
        jLabel7.setToolTipText(stringDatabase.getString("Learning.Alpha.Tooltip"));
        kValueLabel.setText(stringDatabase.getString("Learning.KDB.kValue") + ":");
        kValueLabel.setToolTipText(stringDatabase.getString("Learning.KDB.kValue.Tooltip"));
        kValueField.setText(String.valueOf(kValue));

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
                                                .addComponent(kValueLabel).addComponent(jLabel7))
                                                       .addPreferredGap(LayoutStyle.ComponentPlacement.UNRELATED).addGroup(
                                                jPanel1Layout
                                                        .createParallelGroup(GroupLayout.Alignment.TRAILING)
                                                        .addComponent(kValueField,
                                                                GroupLayout.PREFERRED_SIZE, 40,
                                                                GroupLayout.PREFERRED_SIZE)
                                                        .addComponent(alphaText, GroupLayout.PREFERRED_SIZE,
                                                                40, GroupLayout.PREFERRED_SIZE))))
                        .addContainerGap(12, Short.MAX_VALUE)));
        jPanel1Layout.setVerticalGroup(jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                .addGroup(jPanel1Layout.createSequentialGroup().addGap(18, 18, 18)
                                       .addGroup(jPanel1Layout.createParallelGroup(GroupLayout.Alignment.BASELINE)
                                .addComponent(kValueLabel)
                                .addComponent(kValueField, GroupLayout.PREFERRED_SIZE,
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
                errorMessages.add(stringDatabase.getString("Learning.KDB.IncorrectParameter"));
            }
        } catch (NumberFormatException e) {
            errorMessages.add(stringDatabase.getString("Learning.KDB.IncorrectParameter"));
        }

        try {
            
            int k = Integer.parseInt(kValueField.getText());
            if(k < 0){
                errorMessages.add(stringDatabase.getString("Learning.KDB.KIncorrectParents"));
            }
        } catch (NumberFormatException e) {
            errorMessages.add(stringDatabase.getString("Learning.KDB.KIncorrectParents"));

        }

        if(!errorMessages.isEmpty()){
            JOptionPane.showMessageDialog(null, StringUtil.join(CommonUtils.LINE_SEPARATOR, errorMessages.toArray()),
                        stringDatabase.getString("ErrorWindow.Title.Label"), JOptionPane.ERROR_MESSAGE);
                return;
        }
        
        kValue = Integer.parseInt(kValueField.getText());
        alphaParameter = alphaText.getText();
        this.setVisible(false);
    }



    @Override
    public String getDescription() {
        return StringUtil.join(CommonUtils.LINE_SEPARATOR,
                Arrays.asList(stringDatabase.getString("Learning.KDB.Metrics") + ": "+
                                CommonUtils.getStringFromCamelCaseExpression(unconditionedMetric)+", "+
                                CommonUtils.getStringFromCamelCaseExpression(conditionedMetric),
                              stringDatabase.getString("Learning.Alpha") + ": " + alphaParameter,
                              stringDatabase.getString("Learning.KDB.kValue")+ ": "+kValue
                ).toArray());
    }



    @Override
    public LearningAlgorithm getInstance(ProbNet probNet, CaseDatabase database) {
        return new KDBAlgorithm(probNet, database, getMetricByName(conditionedMetric), getMetricByName(unconditionedMetric),
                                Double.parseDouble(alphaParameter), Integer.parseInt(kValueField.getText()));
    }

    @Override
    public ArrayList<Object> getOptions(){
        ArrayList<Object> options=new ArrayList<>();
        options.add(getMetricByName(conditionedMetric));
        options.add(getMetricByName(unconditionedMetric));
        options.add(Double.parseDouble(alphaParameter));
        options.add(Integer.valueOf(kValueField.getText()));
        return options;
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

    public String getMetric() {
        return unconditionedMetric;
    }

}
