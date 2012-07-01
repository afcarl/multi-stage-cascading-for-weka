/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package multistagecascadingforweka;

import java.awt.BorderLayout;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.MultiStageCascading;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.NBTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
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
 * @author Ivan Mushketyk <ivan.mushketik at gmail.com>
 */
public class MultiStageCascadingForWeka {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        MultiStageCascading msc = new MultiStageCascading();
        J48 classifier1 = new J48();
        IBk knn = new IBk(3);
        
        msc.setClassifiers(new Classifier[] {classifier1, new NBTree()});
        msc.setDebug(true);
        msc.setKNNClassifier(knn);
        
        
        msc.setThresholds("0.95,0.97");
        msc.setPercentTrainingInstances(0.8);
        
        Instances dataset = DataSource.read("D:\\Datasets\\ImageSpam\\WorkFiles\\ARFFs\\textRegionBasedFeatures.arff");
        dataset.setClassIndex(dataset.numAttributes() - 1);
        
        Pair<Instances, Instances> sets = seprateTestAndTrainingSets(dataset, 0.7);
        Instances trainingSet = sets.getFirst();
        Instances testSet = sets.getSecond();
        
        
        msc.buildClassifier(trainingSet);
        
        // display classifier
        final javax.swing.JFrame jf =
                new javax.swing.JFrame("Weka Classifier Tree Visualizer: J48");
        jf.setSize(500, 400);
        jf.getContentPane().setLayout(new BorderLayout());
        TreeVisualizer tv = new TreeVisualizer(null,
                classifier1.graph(),
                new PlaceNode2());
        jf.getContentPane().add(tv, BorderLayout.CENTER);
        jf.addWindowListener(new java.awt.event.WindowAdapter() {

            public void windowClosing(java.awt.event.WindowEvent e) {
                jf.dispose();
            }
        });

        jf.setVisible(true);
        tv.fitToScreen();
        
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
