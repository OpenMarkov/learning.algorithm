open module org.openmarkov.learning.algorithm {
    requires java.desktop;
    requires org.apache.logging.log4j;
    requires org.apache.poi.poi;

    requires org.openmarkov.core;
    requires org.openmarkov.gui;
    requires org.openmarkov.io;
    requires org.openmarkov.inference;
    requires org.openmarkov.learning.core;
    requires org.openmarkov.learning.gui;
    requires org.openmarkov.learning.metric;
    
    exports org.openmarkov.learning.algorithm.scoreAndSearch;
    
    exports org.openmarkov.learning.algorithm.pc;
    exports org.openmarkov.learning.algorithm.hillclimbing;
    exports org.openmarkov.learning.algorithm.naivebayes;
    exports org.openmarkov.learning.algorithm.hillclimbing.gui;
    exports org.openmarkov.learning.algorithm.nbderived.snb.gui;
    exports org.openmarkov.learning.algorithm.nbderived.spnb.gui;
    exports org.openmarkov.learning.algorithm.nbderived.treeaugmentednb.gui;
    exports org.openmarkov.learning.algorithm.pc.gui;
    exports org.openmarkov.learning.algorithm.em.gui;
    exports org.openmarkov.learning.algorithm.pc.independencetester;
    
}
