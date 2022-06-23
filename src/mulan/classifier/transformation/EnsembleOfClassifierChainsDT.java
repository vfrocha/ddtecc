/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package mulan.classifier.transformation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerGridSearch;
import mulan.classifier.MultiLabelOutput;
import static mulan.classifier.meta.RAkELDT.binomial;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.Measure;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * <p>
 * Implementation of the Ensemble of Classifier Chains(ECC) algorithm.</p>
 * <p>
 * For more information, see <em>Read, J.; Pfahringer, B.; Holmes, G., Frank, E.
 * (2011) Classifier Chains for Multi-label Classification. Machine Learning.
 * 85(3):335-359.</em></p>
 *
 * @author Eleftherios Spyromitros-Xioufis
 * @author Konstantinos Sechidis
 * @author Grigorios Tsoumakas
 * @version 2012.02.27
 */
public class EnsembleOfClassifierChainsDT extends TransformationBasedMultiLabelLearner implements MultiLabelLearnerGridSearch {

    /**
     * The number of classifier chain models
     */
    protected int numOfModels;
    /**
     * Parameter for the threshold of discretization of prediction output
     */
    protected double threshold = 0.5;
    /**
     * /**
     * An array of ClassifierChain models
     */
    protected ClassifierChain[] ensemble;
    /**
     * Random number generator
     */
    protected Random rand;
    /**
     * Whether the output is computed based on the average votes or on the
     * average confidences
     */
    protected boolean useConfidences;
    /**
     * Whether to use sampling with replacement to create the data of the models
     * of the ensemble
     */
    protected boolean useSamplingWithReplacement = true;
    /**
     * The size of each bag sample, as a percentage of the training size. Used
     * when useSamplingWithReplacement is true
     */
    protected int BagSizePercent = 100;
    private MLDT MultiLabelDecisionTemplate;
    private MLDT.PredictionType predictionType;
    private double criticalPhiValue;

    /**
     * Returns the size of each bag sample, as a percentage of the training size
     *
     * @return the size of each bag sample, as a percentage of the training size
     */
    public int getBagSizePercent() {
        return BagSizePercent;
    }

    /**
     * Sets the size of each bag sample, as a percentage of the training size
     *
     * @param bagSizePercent the size of each bag sample, as a percentage of the
     * training size
     */
    public void setBagSizePercent(int bagSizePercent) {
        BagSizePercent = bagSizePercent;
    }

    /**
     * Returns the sampling percentage
     *
     * @return the sampling percentage
     */
    public double getSamplingPercentage() {
        return samplingPercentage;
    }

    /**
     * Sets the sampling percentage
     *
     * @param samplingPercentage the sampling percentage
     */
    public void setSamplingPercentage(double samplingPercentage) {
        this.samplingPercentage = samplingPercentage;
    }
    /**
     * The size of each sample, as a percentage of the training size Used when
     * useSamplingWithReplacement is false
     */
    protected double samplingPercentage = 67;

    /**
     * Default constructor
     */
    public EnsembleOfClassifierChainsDT() {
        this(new J48(), 10, true, true, MLDT.PredictionType.COMBINED);
    }

    /**
     * Creates a new object
     *
     * @param classifier the base classifier for each ClassifierChain model
     * @param aNumOfModels the number of models
     * @param doUseConfidences whether to use confidences or not
     * @param doUseSamplingWithReplacement whether to use sampling with
     * replacement or not
     * @param predictionType
     */
    public EnsembleOfClassifierChainsDT(Classifier classifier, int aNumOfModels,
            boolean doUseConfidences, boolean doUseSamplingWithReplacement, MLDT.PredictionType predictionType) {
        super(classifier);
        numOfModels = aNumOfModels;
        useConfidences = doUseConfidences;
        useSamplingWithReplacement = doUseSamplingWithReplacement;
        ensemble = new ClassifierChain[aNumOfModels];
        rand = new Random(1);
        this.predictionType = predictionType;
    }
    
    /**
     * 
     * @param classifier
     * @param aNumOfModels
     * @param doUseConfidences
     * @param doUseSamplingWithReplacement
     * @param criticalPhiValue 
     */
    public EnsembleOfClassifierChainsDT(Classifier classifier, int aNumOfModels,
            boolean doUseConfidences, boolean doUseSamplingWithReplacement, double criticalPhiValue) {
        super(classifier);
        numOfModels = aNumOfModels;
        useConfidences = doUseConfidences;
        useSamplingWithReplacement = doUseSamplingWithReplacement;
        ensemble = new ClassifierChain[aNumOfModels];
        rand = new Random(1);
        this.predictionType = MLDT.PredictionType.MASK;
        this.criticalPhiValue = criticalPhiValue;
    }

    public EnsembleOfClassifierChainsDT(Classifier classifier, int aNumOfModels,
            boolean doUseConfidences, boolean doUseSamplingWithReplacement, MLDT.PredictionType predictionType, double threshold) {
        super(classifier);
        numOfModels = aNumOfModels;
        useConfidences = doUseConfidences;
        useSamplingWithReplacement = doUseSamplingWithReplacement;
        ensemble = new ClassifierChain[aNumOfModels];
        rand = new Random(1);
        this.predictionType = predictionType;
        this.threshold = threshold;
    }

    @Override
    public MultiLabelLearnerGridSearch gridSearch(MultiLabelInstances train, MultiLabelInstances validation) throws InvalidDataFormatException {
        List<Measure> measures = new ArrayList<Measure>(1);
        measures.add(new ExampleBasedFMeasure());
        Evaluator eval = new Evaluator();
        double best = 0;
        Classifier[] baseLearners = {new J48(), new NaiveBayes()};
        List<MLDT.PredictionType> mldtTypes = new ArrayList<>();//{MLDT.PredictionType.INDIVIDUAL, MLDT.PredictionType.COMBINED};
        boolean[] booleans = {true, false};

        if (predictionType != null) {
            mldtTypes.add(predictionType);
        } else {
            mldtTypes.add(MLDT.PredictionType.INDIVIDUAL);
            mldtTypes.add(MLDT.PredictionType.COMBINED);
            mldtTypes.add(MLDT.PredictionType.THRESHOLDING);
        }
        double[] confs = {0.3, 0.5, 0.7};
        if(predictionType == MLDT.PredictionType.MAJT){
            confs = new double[]{0.5};
        }

        for (double conf : confs) {
            for (Classifier baseLearner1 : baseLearners) {
                for (boolean confidences : booleans) {
                    for (boolean replacement : booleans) {
                        for (MLDT.PredictionType mldtType : mldtTypes) {
                            try {
                                EnsembleOfClassifierChainsDT ensembleOfClassifierChainsDT = new EnsembleOfClassifierChainsDT(baseLearner1, this.numOfModels, confidences, replacement, mldtType, conf);
                                ensembleOfClassifierChainsDT.build(train);
                                Evaluation evaluate = eval.evaluate(ensembleOfClassifierChainsDT, validation, measures);
                                double mean = evaluate.getValue(0);
                                if (mean > best) {
                                    best = mean;
                                    this.baseClassifier = baseLearner1;
                                    this.useConfidences = confidences;
                                    this.useSamplingWithReplacement = replacement;
                                    this.predictionType = mldtType;
                                    this.threshold = conf;
                                }
                            } catch (Exception ex) {
                                Logger.getLogger(EnsembleOfClassifierChainsDT.class.getName()).log(Level.SEVERE, null, ex);
                            }
                        }
                    }
                }
            }
        }
//        System.out.print(threshold + ";");
//        System.out.println("best prediction>>" + predictionType);
//        System.out.println("best>>" + baseClassifier + "," + useConfidences + "," + useSamplingWithReplacement + "," + predictionType);
        return new EnsembleOfClassifierChainsDT(baseClassifier, this.numOfModels, useConfidences, useSamplingWithReplacement, predictionType, threshold);
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {

        Instances dataSet = new Instances(trainingSet.getDataSet());

        // default number of models = twice the number of labels
        if (numOfModels == 0) {
            numOfModels = Math.min(2 * numLabels, binomial(numLabels, 3));
        }
        
        for (int i = 0; i < numOfModels; i++) {
            debug("ECC Building Model:" + (i + 1) + "/" + numOfModels);
            Instances sampledDataSet;
            dataSet.randomize(rand);
            if (useSamplingWithReplacement) {
                int bagSize = dataSet.numInstances() * BagSizePercent / 100;
                // create the in-bag dataset
                sampledDataSet = dataSet.resampleWithWeights(new Random(1));
                if (bagSize < dataSet.numInstances()) {
                    sampledDataSet = new Instances(sampledDataSet, 0, bagSize);
                }
            } else {
                RemovePercentage rmvp = new RemovePercentage();
                rmvp.setInvertSelection(true);
                rmvp.setPercentage(samplingPercentage);
                rmvp.setInputFormat(dataSet);
                sampledDataSet = Filter.useFilter(dataSet, rmvp);
            }
            MultiLabelInstances train = new MultiLabelInstances(sampledDataSet, trainingSet.getLabelsMetaData());

            int[] chain = new int[numLabels];
            for (int j = 0; j < numLabels; j++) {
                chain[j] = j;
            }
            for (int j = 0; j < chain.length; j++) {
                int randomPosition = rand.nextInt(chain.length);
                int temp = chain[j];
                chain[j] = chain[randomPosition];
                chain[randomPosition] = temp;
            }
            debug(Arrays.toString(chain));

            // MAYBE WE SHOULD CHECK NOT TO PRODUCE THE SAME VECTOR FOR THE
            // INDICES
            // BUT IN THE PAPER IT DID NOT MENTION SOMETHING LIKE THAT
            // IT JUST SIMPLY SAY A RANDOM CHAIN ORDERING OF L
            ensemble[i] = new ClassifierChain(baseClassifier, chain);
            ensemble[i].build(train);
        }
        MultiLabelDecisionTemplate = new MLDT(ensemble, predictionType, threshold);
        MultiLabelDecisionTemplate.setBuilEach(false);
        MultiLabelDecisionTemplate.setCriticalPhiValue(criticalPhiValue);
        MultiLabelDecisionTemplate.build(trainingSet);
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception,
            InvalidDataException {

//        int[] sumVotes = new int[numLabels];
//        double[] sumConf = new double[numLabels];
//
//        Arrays.fill(sumVotes, 0);
//        Arrays.fill(sumConf, 0);
        return MultiLabelDecisionTemplate.makePredictionInternal(instance);

//        for (int i = 0; i < numOfModels; i++) {
//            MultiLabelOutput ensembleMLO = ensemble[i].makePrediction(instance);
//            boolean[] bip = ensembleMLO.getBipartition();
//            double[] conf = ensembleMLO.getConfidences();
//
//            for (int j = 0; j < numLabels; j++) {
//                sumVotes[j] += bip[j] == true ? 1 : 0;
//                sumConf[j] += conf[j];
//            }
//        }
//
//        double[] confidence = new double[numLabels];
//        for (int j = 0; j < numLabels; j++) {
//            if (useConfidences) {
//                confidence[j] = sumConf[j] / numOfModels;
//            } else {
//                confidence[j] = sumVotes[j] / (double) numOfModels;
//            }
//        }
//
//        MultiLabelOutput mlo = new MultiLabelOutput(confidence, 0.5);
//        return mlo;
    }
}
