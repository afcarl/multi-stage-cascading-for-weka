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
 * along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
 */
package weka.classifiers.meta;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;
import weka.classifiers.Classifier;
import weka.classifiers.MultipleClassifiersCombiner;
import weka.classifiers.RandomizableMultipleClassifiersCombiner;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.filters.Filter;

/**
 * <!-- globalinfo-start --> <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start -- <!-- technical-bibtex-end -->
 *
 * <!-- options-start --> <!-- options-end -->
 *
 * @author Ivan Mushketyk <ivan.mushketik at gmail.com>
 * @version 0.1
 */
public class MultiStageCascading extends RandomizableMultipleClassifiersCombiner
        implements TechnicalInformationHandler {

    /**
     * for serialization
     */
    static final long serialVersionUID = 3724314652175299374L;
    private double[] thresholds = getDefaultProbabilities(1);
    private Classifier kNNClassifier = getDefaultKNN();
    private double[] selectProbabilities;
    private Instances trainInstances;
    private Instances kNNTrainingInstances;
    private Random random = new Random(getSeed());
    private double percentTrainingInstances = getDefaultPercentage();
    boolean lastClassifierChanged = false;

    @Override
    public void buildClassifier(Instances dataset) throws Exception {

        if (getClassifiers().length != this.thresholds.length) {
            throw new Exception("Number of thresholds should be equal to the number of classifers");
        }

        getCapabilities().testWithFail(dataset);

        divideInstances(dataset);
        initializeInstancesProbabilities();

        for (Classifier classifier : this.getClassifiers()) {
            Instances instancesForClassifier = selectInstances();

            if (getDebug()) {
                System.out.println("Training classifer " + classifier.getClass());
                System.out.println("Number of instances for classifer " + instancesForClassifier.numInstances());
            }

            classifier.buildClassifier(instancesForClassifier);
            updateInstancesProbabilities(classifier);
        }

        addInstancesForKNN();
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        Classifier[] classifiers = getClassifiers();
        for (int i = 0; i < classifiers.length; i++) {
            Classifier classifier = classifiers[i];
            double[] distribution = classifier.distributionForInstance(instance);
            double confidence = getConfidence(distribution);
            if (classifierIsConfident(confidence, i)) {
                if (getDebug()) {
                    System.out.println("Classifier number " + i + " is confident");
                    System.out.println("Classifier's confidence " + confidence);
                }
                return distribution;
            }
        }

        if (getDebug()) {
            System.out.println("Using kNN classifier");
        }

        return kNNClassifier.distributionForInstance(instance);
    }

    public Enumeration listOptions() {

        Vector newVector = new Vector();

        newVector.addElement(new Option(
                "\tThresholds of confidence for classifiers.",
                "T", 1, "-T <thresholds for classifers>"));

        newVector.addElement(new Option(
                "\tPercentage of training instances that will be used for training each classifier.",
                "P", 1, "-P <thresholds for classifers>"));

        newVector.addElement(new Option(
                "\tkNN classifier that will be used if all user-specified classifiers are not confident in their predictions.",
                "K", 1, "-K <IBk classifier>"));

        Enumeration enu = super.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }

        return newVector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {

        String kNNString = Utils.getOption('K', options);
        if (kNNString.length() > 0) {
            String[] kNNSpec = Utils.splitOptions(kNNString);
            if (kNNSpec.length == 0) {
                throw new IllegalArgumentException("Invalid filter specification string");
            }
            String kNNName = kNNSpec[0];
            kNNSpec[0] = "";
            setKNNClassifier((IBk) Utils.forName(Filter.class, kNNName, kNNSpec));
        } else {
            setKNNClassifier(getDefaultKNN());
        }

        String percentTrainingSet = Utils.getOption("P", options);
        if (percentTrainingSet.length() != 0) {
            this.percentTrainingInstances = Double.parseDouble(percentTrainingSet);
        } else {
            this.percentTrainingInstances = getDefaultPercentage();
        }
        
        String thresholdsStr = Utils.getOption("T", options);
        if (percentTrainingSet.length() !=0) {
            setThresholds(thresholdsStr);
        } else {
            this.thresholds = getThresholds();
        }

        super.setOptions(options);

        Utils.checkForRemainingOptions(options);
    }

    public String globalInfo() {
        return "MultiStage Cascading classifier use several consequent classifers "
                + "to predict a class of an instance. The next classifier is used "
                + "only if previous classifier is not confident. If all classifiers "
                + "are not confident kNN classifier is used to classify the instance";
    }
    
    @Override
    public String[] getOptions() {
        Vector result;
        String[] options;
        int i;

        result = new Vector();

        if (thresholdChanged()) {
            result.add("-T");
            result.add("" + Utils.arrayToString(getThresholds()));
        }

        if (percentTrainingInstnacesChanged()) {
            result.add("-P");
            result.add("" + getPercentTrainingInstances());
        }
        
        if (lastClassifierChanged()) {
            result.add("-K");
            result.add("" + kNNClassifier.getClass().getName() + " " + Utils.joinOptions(((OptionHandler)kNNClassifier).getOptions()));
        }

        options = super.getOptions();
        for (i = 0; i < options.length; i++) {
            result.add(options[i]);
        }

        return (String[]) result.toArray(new String[result.size()]);
    }
    
    private boolean thresholdChanged() {
        return this.thresholds.length != 0 && this.thresholds[0] != 0.5;
    }

    private boolean percentTrainingInstnacesChanged() {
        return this.percentTrainingInstances != getDefaultPercentage();
    }

    private boolean lastClassifierChanged() {
        return this.lastClassifierChanged;
    }

    @Override
    public void setClassifiers(Classifier[] classifiers) {
        super.setClassifiers(classifiers);
        setThresholds(getDefaultProbabilities(classifiers.length));
    }
    
    public void setThresholds(String strThresholds) {
        String[] thresholdsStr = strThresholds.split(",");
        this.thresholds = new double[thresholdsStr.length];
        
        for (int i = 0; i < thresholdsStr.length; i++) {
            thresholds[i] = Double.parseDouble(thresholdsStr[i]);
        }
    }
    
    protected void setThresholds(double[] thresholds) {
        this.thresholds = thresholds;
    }

    public double[] getThresholds() {
        return this.thresholds;
    }

    public String thresholdsTipText() {
        return "Confidence level for each of the classifiers. First threshold is a confidence leverl"
                + "for the first classifier.";
    }

    public Classifier getKNNClassifier() {
        return this.kNNClassifier;
    }

    public void setKNNClassifier(Classifier kNNClassifier) {
        this.kNNClassifier = kNNClassifier;
        this.lastClassifierChanged = true;
    }

    public String KNNClassifierTipText() {
        return "kNN classifier that will be used by cascading algorithm if non of "
                + "the user specified classifers are confident in their decision";
    }

    public double getPercentTrainingInstances() {
        return this.percentTrainingInstances;
    }

    public void setPercentTrainingInstances(double percentTrainingInstances) {
        this.percentTrainingInstances = percentTrainingInstances;
    }

    public String percentTrainingInstancesTipText() {
        return "Percent of training data that will be used to train each classifier";
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        return result;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation technicalInformation = new TechnicalInformation(TechnicalInformation.Type.CONFERENCE);

        technicalInformation.setValue(TechnicalInformation.Field.AUTHOR, "Cenk Kaynak, Ethem Alpaydin");
        technicalInformation.setValue(TechnicalInformation.Field.YEAR, "2000");
        technicalInformation.setValue(TechnicalInformation.Field.TITLE, "MultiStage Cascading of Multiple Classifiers: One Man's Noise is Another Man's Data");
        technicalInformation.setValue(TechnicalInformation.Field.ORGANIZATION, "Proceedings of the 25th international conference on Machine learning (2000) ");
        technicalInformation.setValue(TechnicalInformation.Field.PAGES, "455-462");

        return technicalInformation;
    }

    private void divideInstances(Instances instances) {
        this.trainInstances = new Instances(instances, 0, 0);
        this.kNNTrainingInstances = new Instances(instances, 0, 0);

        for (int i = 0; i < instances.numInstances(); i++) {
            Instance instance = instances.instance(i);
            if (this.random.nextDouble() >= 0.5) {
                this.trainInstances.add(instance);
            } else {
                this.kNNTrainingInstances.add(instance);
            }
        }
    }

    private void initializeInstancesProbabilities() {

        this.selectProbabilities = new double[trainInstances.numInstances()];

        double initProbability = 1.0 / selectProbabilities.length;
        for (int i = 0; i < this.selectProbabilities.length; i++) {
            this.selectProbabilities[i] = initProbability;
        }
    }

    private Instances selectInstances() {
        Instances selectedInstances = new Instances(trainInstances, 0, 0);

        int prevInstance = -1;
        int currentInstance = 0;
        int numTrainingInstances = (int) (this.trainInstances.numInstances() * percentTrainingInstances);
        for (int i = 0; i < numTrainingInstances; i++) {
            double step = this.random.nextDouble();

            while (step >= 0) {
                step -= this.selectProbabilities[currentInstance];
                prevInstance = currentInstance;
                currentInstance = (currentInstance + 1) % this.selectProbabilities.length;
            }

            selectedInstances.add(trainInstances.instance(prevInstance));
        }

        return selectedInstances;
    }

    private void updateInstancesProbabilities(Classifier classifier) throws Exception {

        double sum = 0;
        for (int i = 0; i < trainInstances.numInstances(); i++) {
            Instance instance = trainInstances.instance(i);

            double[] distribution = classifier.distributionForInstance(instance);
            double maxProbability = getConfidence(distribution);
            this.selectProbabilities[i] = maxProbability;
            sum += maxProbability;
        }

        for (int i = 0; i < this.selectProbabilities.length; i++) {
            this.selectProbabilities[i] /= sum;
        }
    }

    private void addInstancesForKNN() throws Exception {

        Instances kNNInstances = new Instances(this.kNNTrainingInstances, 0, 0);
        for (int i = 0; i < this.kNNTrainingInstances.numInstances(); i++) {
            Instance instance = this.kNNTrainingInstances.instance(i);

            boolean confidentClassifierFound = false;
            Classifier[] classifiers = getClassifiers();
            for (int ci = 0; ci < classifiers.length; ci++) {
                Classifier classifier = classifiers[ci];
                double[] distribution = classifier.distributionForInstance(instance);

                double confidence = getConfidence(distribution);
                if (classifierIsConfident(confidence, ci)) {
                    confidentClassifierFound = true;
                    break;
                }
            }

            if (!confidentClassifierFound) {
                kNNInstances.add(instance);
            }
        }

        this.kNNClassifier.buildClassifier(kNNInstances);
    }

    private boolean classifierIsConfident(double confidence, int classifierIndex) {
        return confidence > this.thresholds[classifierIndex];
    }

    private double getConfidence(double[] distribution) {
        double maxProbability = -1;

        for (double probability : distribution) {
            if (probability > maxProbability) {
                maxProbability = probability;
            }
        }

        return maxProbability;
    }

    private IBk getDefaultKNN() {
        return new IBk();
    }

    private double[] getDefaultProbabilities() {
        return getDefaultProbabilities(getClassifiers().length);
    }
    
    private double[] getDefaultProbabilities(int arrSize) {
        double[] probabilities = new double[arrSize];
        
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = 0.5;
        }
        
        return probabilities;
    }

    private double getDefaultPercentage() {
        return 0.8;
    }

   
}
