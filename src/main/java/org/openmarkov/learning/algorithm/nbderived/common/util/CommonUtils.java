package org.openmarkov.learning.algorithm.nbderived.common.util;

import org.apache.poi.util.StringUtil;

public class CommonUtils {

    private final static String REG_EXP_CAMEL_CASE = "(?<!^)(?=[A-Z])";

    public final static String LINE_SEPARATOR = System.getProperty("line.separator");



    public static String getStringFromCamelCaseExpression(String exp){
        String result = StringUtil.join(exp.split(REG_EXP_CAMEL_CASE), " ");
        return result.length()>1? (result.substring(0, 1).toUpperCase()+result.substring(1, result.length()).toLowerCase()) :
                                    result;
    }



}
