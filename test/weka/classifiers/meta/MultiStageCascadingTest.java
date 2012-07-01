/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.meta;

import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;

import junit.framework.Test;
import junit.framework.TestSuite;

/**
 *
 * @author Ivan Mushketyk <ivan.mushketik at gmail.com>
 */
public class MultiStageCascadingTest extends AbstractClassifierTest {
    
     public MultiStageCascadingTest(String name) { super(name);  }

  /** Creates a default J48 */
    public Classifier getClassifier() {
        return new MultiStageCascading();
    } 
}
