open module org.openmarkov.learning.algorithm {
    requires transitive java.desktop;
    requires org.apache.logging.log4j;
    requires org.apache.poi.poi;

    requires transitive org.openmarkov.core;
    requires org.openmarkov.inference;
    requires transitive org.openmarkov.learning.core;
    requires org.openmarkov.learning.metric;
    requires org.jetbrains.annotations;
    
    
    exports org.openmarkov.learning.algorithm.em;
    exports org.openmarkov.learning.algorithm.hillclimbing;
    exports org.openmarkov.learning.algorithm.naivebayes;
    exports org.openmarkov.learning.algorithm.nbderived.common;
    exports org.openmarkov.learning.algorithm.nbderived.fanb;
    exports org.openmarkov.learning.algorithm.nbderived.kdb;
    exports org.openmarkov.learning.algorithm.nbderived.snb;
    exports org.openmarkov.learning.algorithm.nbderived.spnb;
    exports org.openmarkov.learning.algorithm.nbderived.treeaugmentednb;
    exports org.openmarkov.learning.algorithm.pc;
    exports org.openmarkov.learning.algorithm.pc.independencetester;
    exports org.openmarkov.learning.algorithm.scoreAndSearch;
    
}
