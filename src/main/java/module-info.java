module org.openmarkov.learning.algorithm {
    requires org.openmarkov.core;
    requires org.openmarkov.learning.core;
    requires java.desktop;
    requires org.openmarkov.gui;
    requires org.openmarkov.learning.gui;
    requires org.apache.logging.log4j;
    requires org.openmarkov.io;
    requires org.openmarkov.inference;
    
    exports org.openmarkov.learning.algorithm.scoreAndSearch;
    exports org.openmarkov.learning.algorithm.scoreAndSearch.metric;
    exports org.openmarkov.learning.algorithm.scoreAndSearch.metric.annotation;
    exports org.openmarkov.learning.algorithm.scoreAndSearch.metric.util;
    
    exports org.openmarkov.learning.algorithm.pc;
    exports org.openmarkov.learning.algorithm.hillclimbing;
    
}
