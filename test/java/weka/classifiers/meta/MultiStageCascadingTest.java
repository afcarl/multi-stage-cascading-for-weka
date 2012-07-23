/*
 *This file is part of MultiStageCascading for Weka.
 * Foobar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MultiStageCascading for Weka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MultiStageCascading for Weka.  If not, see <http://www.gnu.org/licenses/>.
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
