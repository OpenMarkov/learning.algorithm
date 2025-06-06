module org.openmarkov.learning.algorithm {
    requires java.desktop;
    requires org.apache.logging.log4j;
    requires org.apache.poi.poi;

    requires org.openmarkov.core;
    requires org.openmarkov.gui;
    requires org.openmarkov.io;
    requires org.openmarkov.inference;
    requires org.openmarkov.learning.core;
    requires org.openmarkov.learning.gui;
    
    exports org.openmarkov.learning.algorithm.scoreAndSearch;
    exports org.openmarkov.learning.algorithm.scoreAndSearch.metric;
    exports org.openmarkov.learning.algorithm.scoreAndSearch.metric.annotation;
    exports org.openmarkov.learning.algorithm.scoreAndSearch.metric.util;
    
    exports org.openmarkov.learning.algorithm.pc;
    exports org.openmarkov.learning.algorithm.hillclimbing;
    exports org.openmarkov.learning.algorithm.scoreAndSearch.cache;
    exports org.openmarkov.learning.algorithm.naivebayes;
    
}
