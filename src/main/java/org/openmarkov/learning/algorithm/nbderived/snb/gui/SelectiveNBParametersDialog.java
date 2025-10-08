package org.openmarkov.learning.algorithm.nbderived.snb.gui;

import org.apache.poi.util.StringUtil;
import org.jetbrains.annotations.Nullable;
import org.openmarkov.core.exception.InvalidArgumentException;
import org.openmarkov.core.exception.UnreacheableException;
import org.openmarkov.core.exception.UnrecoverableException;
import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.learning.algorithm.nbderived.snb.SelectiveNBAlgorithm;
import org.openmarkov.learning.metric.Metric;
import org.openmarkov.learning.metric.annotation.MetricManager;
import org.openmarkov.learning.core.algorithm.LearningAlgorithm;
import org.openmarkov.learning.gui.AlgorithmConfiguration;
import org.openmarkov.learning.gui.AlgorithmParametersDialog;

import static org.openmarkov.learning.algorithm.nbderived.common.util.CommonUtils.LINE_SEPARATOR;
import static org.openmarkov.learning.algorithm.nbderived.common.util.CommonUtils.getStringFromCamelCaseExpression;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


@SuppressWarnings("serial")
@AlgorithmConfiguration(algorithm = SelectiveNBAlgorithm.class)
public class SelectiveNBParametersDialog extends AlgorithmParametersDialog {
    
    private String metric = "Accuracy";
    private static String alphaParameter = "0.5";
    private static String significanceLevel = "0.05";
    private MetricManager metricManager;
    
    private JButton AcceptButton;
    private JLabel testerLabel;
    private JTextField alphaText;
    private JTextField significanceLevelText;
    private JLabel jLabel7;
    private JLabel jLabelForward;
    private JPanel jPanel1;
    private JCheckBox forwardCheckbox;
    
    public SelectiveNBParametersDialog(JFrame parent, boolean modal) {
        super(parent, modal);
        setLocationRelativeTo(parent);
        metricManager = new MetricManager();
        initComponents();
    }
    
    @Override
    public String getDescription() {
        return StringUtil.join(LINE_SEPARATOR,
                               Arrays.asList(stringDatabase.getString("Learning.SelectiveNaiveBayes.Metric") + ": " +
                                                     getStringFromCamelCaseExpression(metric),
                                             stringDatabase.getString("Learning.Alpha") + ": " + alphaParameter
                               ).toArray());
    }
    
    
    @Override
    public LearningAlgorithm getInstance(ProbNet probNet, CaseDatabase database) {
        try {
            Metric metricInstance = (Metric) Arrays.stream(metricManager.getMetricByName(metric).getConstructors())
                                                   .iterator()
                                                   .next()
                                                   .newInstance();
            return new SelectiveNBAlgorithm(probNet, database, metricInstance,
                                            Double.parseDouble(alphaParameter),
                                            forwardCheckbox.isSelected());
        } catch (InstantiationException | IllegalAccessException | IllegalArgumentException |
                 InvocationTargetException e) {
            throw new UnreacheableException(e);
        }
    }
    
    @Override
    public ArrayList<Object> getOptions() {
        try {
            ArrayList<Object> options = new ArrayList<>();
            Metric metricInstance = (Metric) Arrays.stream(metricManager.getMetricByName(metric).getConstructors())
                                                   .iterator()
                                                   .next()
                                                   .newInstance();
            options.add(metricInstance);
            options.add(Double.parseDouble(alphaParameter));
            options.add(forwardCheckbox.isSelected());
            return options;
        } catch (InstantiationException | IllegalAccessException | IllegalArgumentException |
                 InvocationTargetException e) {
            throw new UnreacheableException(e);
        }
    }
    
    
    public String getMetric() {
        return metric;
    }
    
    
    private void initComponents() {
        jPanel1 = new JPanel();
        alphaText = new JTextField();
        jLabel7 = new JLabel();
        AcceptButton = new JButton();
        jLabelForward = new JLabel();
        forwardCheckbox = new JCheckBox();
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        setTitle(stringDatabase.getString("Learning.SelectiveNaiveBayes.Title"));
        jPanel1.setBorder(BorderFactory.createTitledBorder(stringDatabase.getString("Learning.SelectiveNaiveBayes.Title")));
        alphaText.setText(alphaParameter);
        jLabel7.setText(stringDatabase.getString("Learning.Alpha") + ":");
        jLabel7.setToolTipText(stringDatabase.getString("Learning.Alpha.Tooltip"));
        jLabelForward.setText(stringDatabase.getString("Learning.SelectiveNaiveBayes.Forward") + ":");
        forwardCheckbox.setSelected(true);
        
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
                                                                                                          jPanel1Layout.createSequentialGroup()
                                                                                                                       .addGap(92, 92, 92)
                                                                                                                       .addComponent(AcceptButton))
                                                                                                  .addGroup(jPanel1Layout.createSequentialGroup()
                                                                                                                         .addContainerGap()
                                                                                                                         .addGroup(
                                                                                                                                 jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                                                                                                                                              .addComponent(jLabelForward)
                                                                                                                                              .addComponent(jLabel7))
                                                                                                                         .addPreferredGap(LayoutStyle.ComponentPlacement.UNRELATED)
                                                                                                                         .addGroup(
                                                                                                                                 jPanel1Layout
                                                                                                                                         .createParallelGroup(GroupLayout.Alignment.TRAILING)
                                                                                                                                         .addComponent(forwardCheckbox,
                                                                                                                                                       GroupLayout.PREFERRED_SIZE, 40,
                                                                                                                                                       GroupLayout.PREFERRED_SIZE)
                                                                                                                                         .addComponent(alphaText, GroupLayout.PREFERRED_SIZE,
                                                                                                                                                       40, GroupLayout.PREFERRED_SIZE))))
                                                                             .addContainerGap(12, Short.MAX_VALUE)));
        jPanel1Layout.setVerticalGroup(jPanel1Layout.createParallelGroup(GroupLayout.Alignment.LEADING)
                                                    .addGroup(jPanel1Layout.createSequentialGroup()
                                                                           .addGap(18, 18, 18)
                                                                           .addGroup(jPanel1Layout.createParallelGroup(GroupLayout.Alignment.BASELINE)
                                                                                                  .addComponent(jLabelForward)
                                                                                                  .addComponent(forwardCheckbox, GroupLayout.PREFERRED_SIZE,
                                                                                                                GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                                                                           .addPreferredGap(LayoutStyle.ComponentPlacement.UNRELATED)
                                                                           .addGroup(
                                                                                   jPanel1Layout.createParallelGroup(GroupLayout.Alignment.BASELINE)
                                                                                                .addComponent(jLabel7)
                                                                                                .addComponent(alphaText, GroupLayout.PREFERRED_SIZE,
                                                                                                              GroupLayout.DEFAULT_SIZE,
                                                                                                              GroupLayout.PREFERRED_SIZE))
                                                                           .addGap(11, 11, 11)
                                                                           .addComponent(AcceptButton)
                                                                           .addContainerGap()));
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
