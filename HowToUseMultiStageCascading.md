# Example of using MultiStage Cascading classifier #

Using MultiStage Cascading classifier is simple. You need just to set sequence of classifiers and corresponding confidence thresholds for each classifier. Then, as with other Weka classifiers, you need build a classifier using a training set. After that you can use it to classify unknown instances.

```

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
        

        Instances trainingSet = ...
        Instances testSet = ...
        
        // Build cascade classifier
        msc.buildClassifier(trainingSet);
       
        // Evaluate created classifier
        Evaluation eval = new Evaluation(trainingSet);
        eval.evaluateModel(msc, testSet);
        System.out.println(eval.toSummaryString("\nResults\n\n", false));

```

Details about how this classifier works you can find in the [paper](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.4851) written by the creators of this algorithm:

C. Kaynak and E. Alpaydin, “MultiStage? Cascading of Multiple Classifiers: One Man’s Noise is Another Man's Data,” IN PROCEEDINGS OF THE 17TH INTERNATIONAL CONFERENCE ON MACHINE LEARNING (ICML-2000, pp. 455 - 462, 2000.