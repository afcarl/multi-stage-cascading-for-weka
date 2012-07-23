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
 <!-- globalinfo-start -->
 * MultiStage Cascading classifier use several consequent classifers to predict a class of an instance. The next classifier is used only if previous classifier is not confident. If all classifiers are not confident kNN classifier is used to classify the instance
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;conference{CenkKaynak2000,
 *    author = {Cenk Kaynak, Ethem Alpaydin},
 *    organization = {Proceedings of the 25th international conference on Machine learning (2000) },
 *    pages = {455-462},
 *    title = {MultiStage Cascading of Multiple Classifiers: One Man's Noise is Another Man's Data},
 *    year = {2000}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -T &lt;thresholds for classifers&gt;
 *  Thresholds of confidence for classifiers.</pre>
 * 
 * <pre> -P &lt;thresholds for classifers&gt;
 *  Percentage of training instances that will be used for training each classifier.</pre>
 * 
 * <pre> -K &lt;IBk classifier&gt;
 *  kNN classifier that will be used if all user-specified classifiers are not confident in their predictions.</pre>
 * 
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 * 
 * <pre> -B &lt;classifier specification&gt;
 *  Full class name of classifier to include, followed
 *  by scheme options. May be specified multiple times.
 *  (default: "weka.classifiers.rules.ZeroR")</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 <!-- options-end -->
 *
 * @author Ivan Mushketyk <ivan.mushketik at gmail.com>
 * @version 0.6
 */
public class MultiStageCascading extends RandomizableMultipleClassifiersCombiner
        implements TechnicalInformationHandler {

    /**
     * for serialization
     */
    static final long serialVersionUID = 3724314652175299374L;
    
    // Confidence thresholds for different classifiers, to find out if
    // it should be used for classifying a particular instance
    private double[] confidenceThresholds = getDefaultThresholds(1);
    // Last classifier that is used if all other classifers are not confident
    private Classifier lastClassifier = getDefaultKNN();
    // Probabilities of selecting training instances to train current classifier
    private double[] selectProbabilities;
    // Training instances for the sequence of classifiers
    private Instances trainInstances;
    // Training instances for the last classifier
    private Instances lastClassifierTrainingInstances;
    // Shared random generator with the specified seed for experiments repetability
    private Random random = new Random(getSeed());
    // How many of training instances will be used for training classifiers in a sequence
    private double percentTrainingInstances = getDefaultSelectionPercentage();
    // Flag that shows that last classifier was changed
    boolean lastClassifierChanged = false;

    @Override
    public void buildClassifier(Instances dataset) throws Exception {

        if (getClassifiers().length != this.confidenceThresholds.length) {
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

        trainLastClassifier();
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

        return lastClassifier.distributionForInstance(instance);
    }

    @Override
    public Enumeration listOptions() {

        Vector newVector = new Vector();

        newVector.addElement(new Option(
                "\tThresholds of confidence for classifiers.",
                "T", 1, "-T <thresholds for classifers>"));

        newVector.addElement(new Option(
                "\tPercentage of training instances that will be used for training each classifier.",
                "P", 1, "-P <thresholds for classifers>"));

        newVector.addElement(new Option(
                "\tkNN classifier that will be used if all user-specified classifiers are not confident in their predictions. It is recommended to use kNN for this purposes",
                "K", 1, "-K <IBk classifier>"));

        Enumeration enu = super.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }

        return newVector.elements();
    }
    
    /**
     * Parses a given list of options.
     *
     <!-- options-start -->
     * Valid options are: <p/>
     * 
     * <pre> -T &lt;thresholds for classifers&gt;
     *  Thresholds of confidence for classifiers.</pre>
     * 
     * <pre> -P &lt;thresholds for classifers&gt;
     *  Percentage of training instances that will be used for training each classifier.</pre>
     * 
     * <pre> -K &lt;IBk classifier&gt;
     *  kNN classifier that will be used if all user-specified classifiers are not confident in their predictions.</pre>
     * 
     * <pre> -S &lt;num&gt;
     *  Random number seed.
     *  (default 1)</pre>
     * 
     * <pre> -B &lt;classifier specification&gt;
     *  Full class name of classifier to include, followed
     *  by scheme options. May be specified multiple times.
     *  (default: "weka.classifiers.rules.ZeroR")</pre>
     * 
     * <pre> -D
     *  If set, classifier is run in debug mode and
     *  may output additional info to the console</pre>
     * 
     <!-- options-end -->
     *     
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
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
            setLastClassifier((IBk) Utils.forName(Filter.class, kNNName, kNNSpec));
        } else {
            this.lastClassifier = getDefaultKNN();
        }

        String percentTrainingSet = Utils.getOption("P", options);
        if (percentTrainingSet.length() != 0) {
            this.percentTrainingInstances = Double.parseDouble(percentTrainingSet);
        } else {
            this.percentTrainingInstances = getDefaultSelectionPercentage();
        }
        
        String thresholdsStr = Utils.getOption("T", options);
        if (percentTrainingSet.length() !=0) {
            setConfidenceThresholds(thresholdsStr);
        } else {
            this.confidenceThresholds = getDefaultThresholds();
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

        if (confidenceThresholdsChanged()) {
            result.add("-T");
            result.add("" + Utils.arrayToString(this.confidenceThresholds));
        }

        if (percentTrainingInstnacesChanged()) {
            result.add("-P");
            result.add("" + getPercentTrainingInstances());
        }
        
        if (lastClassifierChanged()) {
            result.add("-K");
            result.add("" + lastClassifier.getClass().getName() + " " + Utils.joinOptions(((OptionHandler)lastClassifier).getOptions()));
        }

        options = super.getOptions();
        for (i = 0; i < options.length; i++) {
            result.add(options[i]);
        }

        return (String[]) result.toArray(new String[result.size()]);
    }
    
    /**
     * Check if thresholds were changed.
     * @return true if this value was changed, false otherwise.
     */
    private boolean confidenceThresholdsChanged() {
        return this.confidenceThresholds.length != 0 && this.confidenceThresholds[0] != 0.5;
    }

    /**
     * Check if percent training instances was changed.
     * @return true if this value was changed, false otherwise
     */
    private boolean percentTrainingInstnacesChanged() {
        return this.percentTrainingInstances != getDefaultSelectionPercentage();
    }

    /**
     * Check if last classifier was changed
     * @return return true if this value was changed, false otherwise.
     */
    private boolean lastClassifierChanged() {
        return this.lastClassifierChanged;
    }

    @Override
    public void setClassifiers(Classifier[] classifiers) {
        super.setClassifiers(classifiers);
        setConfidenceThresholds(getDefaultThresholds(classifiers.length));
    }
    
    /**
     * Set confidence threshold for classifiers
     * @param strThresholds comma separated values of confidence threshold
     * @throws Exception if number of classifiers is not equal to the number of thresholds
     */
    public void setConfidenceThresholds(String strThresholds) throws Exception {   
        
        String[] thresholdsStr = strThresholds.split(",");
        this.confidenceThresholds = new double[thresholdsStr.length];
        
        if (thresholdsStr.length != getClassifiers().length) {
            throw new Exception("Number of threshold is not equals to the number of classifiers");
        }
        
        for (int i = 0; i < thresholdsStr.length; i++) {
            confidenceThresholds[i] = Double.parseDouble(thresholdsStr[i]);
        }
    }
    
    /**
     * Set confidence thresholds for classifiers
     * @param confidenceThresholds 
     */
    protected void setConfidenceThresholds(double[] confidenceThresholds){       
        
        this.confidenceThresholds = confidenceThresholds;
    }

    /**
     * Get string representation of confidence thresholds
     * @return comma separated values of confidence thresholds
     */
    public String getConfidenceThresholds() {
        return Utils.arrayToString(this.confidenceThresholds);
    }

    public String confidenceThresholdsTipText() {
        return "Confidence level for each of the classifiers. First threshold is a confidence leverl"
                + "for the first classifier.";
    }

    /**
     * Get classifier that will be used 
     * @return 
     */
    public Classifier getLastClassifier() {
        return this.lastClassifier;
    }

    /**
     * Set last classifier that will be used if all previous classifiers are not confident
     * @param lastClassifier - classifier that will be used
     */
    public void setLastClassifier(Classifier lastClassifier) {
        this.lastClassifier = lastClassifier;
        this.lastClassifierChanged = true;
    }

    public String lastClassifierTipText() {
        return "kNN classifier that will be used by cascading algorithm if non of "
                + "the user specified classifers are confident in their decision";
    }

    /**
     * Get percent of instances that will be used for training each classifier.
     * @return percent of instances that will be used for training each classifier.
     */
    public double getPercentTrainingInstances() {
        return this.percentTrainingInstances;
    }

    /**
     * Set percent of training instances that will be used to train each classifier.
     * @param percentTrainingInstances 
     */
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

    /**
     * Randomly divide training instances into two categories: training instances
     * for classifiers in sequence and training instances for the last classifier.
     * @param instances - all training instances
     */
    private void divideInstances(Instances instances) {
        this.trainInstances = new Instances(instances, 0, 0);
        this.lastClassifierTrainingInstances = new Instances(instances, 0, 0);

        for (int i = 0; i < instances.numInstances(); i++) {
            Instance instance = instances.instance(i);
            if (this.random.nextDouble() >= 0.5) {
                this.trainInstances.add(instance);
            } else {
                this.lastClassifierTrainingInstances.add(instance);
            }
        }
    }

    /**
     * Initialise probabilities of selecting each training instance
     */
    private void initializeInstancesProbabilities() {

        this.selectProbabilities = new double[trainInstances.numInstances()];

        double initProbability = 1.0 / selectProbabilities.length;
        for (int i = 0; i < this.selectProbabilities.length; i++) {
            this.selectProbabilities[i] = initProbability;
        }
    }

    /**
     * Select instances that will be used for training current classifier.
     * @return training instances for the classifier
     */
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

    /**
     * Update probabilities of selecting instance into a new training set according
     * to classifier's performance.
     * @param classifier - classifier that is used to calculate new probability
     * @throws Exception 
     */
    private void updateInstancesProbabilities(Classifier classifier) throws Exception {

        int classIndex = trainInstances.classIndex();
        
        double sum = 0;
        for (int i = 0; i < trainInstances.numInstances(); i++) {
            Instance instance = trainInstances.instance(i);

            double[] distribution = classifier.distributionForInstance(instance);
                       
            double classValue = instance.value(classIndex);
            double currentClassConfidence = distribution[(int) classValue];
            this.selectProbabilities[i] = currentClassConfidence;
            sum += currentClassConfidence;
        }

        for (int i = 0; i < this.selectProbabilities.length; i++) {
            this.selectProbabilities[i] /= sum;
        }
    }

    /**
     * Train the last classifier.
     * @throws Exception - if training failed.
     */
    private void trainLastClassifier() throws Exception {

        int classIndex = trainInstances.classIndex();
        
        Instances kNNInstances = new Instances(this.lastClassifierTrainingInstances, 0, 0);
        for (int i = 0; i < this.lastClassifierTrainingInstances.numInstances(); i++) {
            Instance instance = this.lastClassifierTrainingInstances.instance(i);

            boolean confidentClassifierFound = false;
            Classifier[] classifiers = getClassifiers();
            for (int ci = 0; ci < classifiers.length; ci++) {
                Classifier classifier = classifiers[ci];
                double[] distribution = classifier.distributionForInstance(instance);

                double classValue = instance.value(classIndex);
                double currentClassConfidence = distribution[(int) classValue];
                if (classifierIsConfident(currentClassConfidence, ci)) {
                    confidentClassifierFound = true;
                    break;
                }
            }

            if (!confidentClassifierFound) {
                kNNInstances.add(instance);
            }
        }

        this.lastClassifier.buildClassifier(kNNInstances);
    }
    /**
     * Check if current classifier is confident
     * @param confidence - confidence of classifier during classifying an instance
     * @param classifierIndex - number of classifier in a sequence
     * @return true if the classifier is confident, false otherwise
     */
    private boolean classifierIsConfident(double confidence, int classifierIndex) {
        return confidence > this.confidenceThresholds[classifierIndex];
    }

    /**
     * Get confidence of current classifier.
     * @param distribution - distribution of probabilities for the classified instance
     * for the current classifier.
     * @return confidence value of the classifier.
     */
    private double getConfidence(double[] distribution) {
        double maxProbability = -1;

        for (double probability : distribution) {
            if (probability > maxProbability) {
                maxProbability = probability;
            }
        }

        return maxProbability;
    }

    /**
     * Get default kNN classifier that is set as the last classifier.
     * @return default kNN classifier
     */
    private IBk getDefaultKNN() {
        return new IBk();
    }

    /**
     * Get default values for classifier's confidence thresholds.
     * @return 
     */
    private double[] getDefaultThresholds() {
        return getDefaultThresholds(getClassifiers().length);
    }
    
    /**
     * Get default values of confidence thresholds.
     * @param arrSize - size of the confidence thresholds array
     * @return default confidence threshold array.
     */
    private double[] getDefaultThresholds(int arrSize) {
        double[] probabilities = new double[arrSize];
        
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = 0.5;
        }
        
        return probabilities;
    }

    /**
     * Get default value of percent of selected instances for training
     * @return percent of instances that will be selected for training.
     */
    private double getDefaultSelectionPercentage() {
        return 0.8;
    }
}

