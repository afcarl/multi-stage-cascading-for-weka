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
package examples;

import java.awt.BorderLayout;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.MultiStageCascading;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.NBTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;


class Pair<F, S> {
    private F first;
    private S second;
    
    public Pair(F first, S second) {
        this.first = first;
        this.second = second;
    }
    
    public F getFirst() {
        return first;
    }
    
    public S getSecond() {
        return second;
    }
}

/**
 *
 * Example of how MultiStage Cascading can be used in your code.
 * @author Ivan Mushketyk <ivan.mushketik at gmail.com>
 */
public class Example {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        
        if (args.length != 1) {
            System.out.println("Requires path to the dataset as the first and only argument");
            return;
        }
        
        final String datasetPath = args[0];
        
        // Create classifiers
        MultiStageCascading msc = new MultiStageCascading();
        J48 classifier1 = new J48();
        IBk knn = new IBk(3);
        
        // Set sequence of classifiers
        msc.setClassifiers(new Classifier[] {classifier1, new NBTree()});
        msc.setDebug(true);
        // Set a classifier that will classify an instance that is not classified by all other classifiers
        msc.setLastClassifier(knn);        

        // First classifier will have confidence threshold 0.95 and the second one 0.97
        msc.setConfidenceThresholds("0.95,0.97");
        // 80% of instances in training set will be randomly selected to train j-th classifier
        msc.setPercentTrainingInstances(0.8);
        
        
        Instances dataset = DataSource.read(datasetPath);
        dataset.setClassIndex(dataset.numAttributes() - 1);
        
        // Create test and training sets
        Pair<Instances, Instances> sets = seprateTestAndTrainingSets(dataset, 0.7);
        Instances trainingSet = sets.getFirst();
        Instances testSet = sets.getSecond();
        
        // Build cascade classifier
        msc.buildClassifier(trainingSet);
       
        // Evaluate created classifier
        Evaluation eval = new Evaluation(trainingSet);
        eval.evaluateModel(msc, testSet);
        System.out.println(eval.toSummaryString("\nResults\n\n", false));
    }
    
    public static Pair<Instances, Instances> seprateTestAndTrainingSets(Instances instances, double probability) {
        Instances trainingSet = new Instances(instances, 0, 0);
        Instances testSet = new Instances(instances, 0, 0);
        Random rand = new Random();
        rand.setSeed(1L);

        for (int i = 0; i < instances.numInstances(); i++) {
            Instance instance = instances.instance(i);
            if (rand.nextDouble() > probability) {
                testSet.add(instance);
            } else {
                trainingSet.add(instance);
            }
        }

        return new Pair<Instances, Instances>(trainingSet, testSet);
    }
}
