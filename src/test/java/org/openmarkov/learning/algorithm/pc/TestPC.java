import org.openmarkov.core.io.database.CaseDatabase;
import org.openmarkov.core.model.network.NodeType;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.action.base.PNEdit;
import org.openmarkov.learning.algorithm.pc.PCAlgorithm;
import org.openmarkov.learning.algorithm.pc.independencetester.IndependenceTester;
import org.openmarkov.learning.core.util.LearningEditProposal;
import org.openmarkov.core.model.network.Node;
import java.util.List;
import java.util.Collections;

public class TestPC {
    public static void main(String[] args) throws Exception {
        ProbNet probNet = new ProbNet();
        Variable varA = new Variable("A");
        Variable varB = new Variable("B");
        Variable varC = new Variable("C");
        Node nodeA = probNet.addNode(varA, NodeType.CHANCE);
        Node nodeB = probNet.addNode(varB, NodeType.CHANCE);
        Node nodeC = probNet.addNode(varC, NodeType.CHANCE);

        probNet.addLink(nodeA, nodeB, false);
        probNet.addLink(nodeB, nodeC, false);
        probNet.addLink(nodeA, nodeC, false);

        IndependenceTester tester = new IndependenceTester() {
            @Override
            public double test(CaseDatabase caseDatabase, Node node1, Node node2, List<Node> adjacencySubset) {
                // Make A and C independent given empty set so their link is removed
                if ((node1.getName().equals("A") && node2.getName().equals("C")) ||
                    (node1.getName().equals("C") && node2.getName().equals("A"))) {
                    return 0.5; // p-value > alpha
                }
                return 0.0;
            }
        };

        PCAlgorithm pc = new PCAlgorithm(probNet, null, 0.05, tester, 0.05);

        System.out.println("Applying getBestEdit");
        LearningEditProposal bestEdit = pc.getBestEdit(false, true);
        System.out.println("First edit: " + (bestEdit != null ? bestEdit.getEdit() : "null"));
        
        if (bestEdit != null) {
            System.out.println("Executing edit...");
            bestEdit.getEdit().executeEdit();
            pc.afterEditExecutes(bestEdit.getEdit()); // Since LearningManager usually calls probNet.doEdit which fires listeners
        }

        System.out.println("\nGetting second edit...");
        bestEdit = pc.getBestEdit(false, true);
        System.out.println("Second edit: " + (bestEdit != null ? bestEdit.getEdit() : "null"));

        while (bestEdit != null) {
            System.out.println("Getting next edit...");
            bestEdit = pc.getNextEdit(false, true);
            System.out.println("Next edit: " + (bestEdit != null ? bestEdit.getEdit() : "null"));
        }
    }
}
