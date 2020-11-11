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

import java.util.Arrays;
import java.util.Random;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import static mulan.classifier.meta.RAkEL.binomial;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * <p>
 * Implementation of the Decition Template Ensemble of Classifier Chains(DTECC) algorithm.</p>
 * <p>
 */
public class EnsembleOfClassifierChainsDT extends TransformationBasedMultiLabelLearner {

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
        this(new J48(), 10, true);
    }

    /**
     * Creates a new object
     *
     * @param classifier the base classifier for each ClassifierChain model
     * @param aNumOfModels the number of models
     * @param doUseSamplingWithReplacement whether to use sampling with
     * replacement or not
     * @param predictionType
     */
    public EnsembleOfClassifierChainsDT(Classifier classifier, int aNumOfModels,
            boolean doUseSamplingWithReplacement) {
        super(classifier);
        numOfModels = aNumOfModels;
        useSamplingWithReplacement = doUseSamplingWithReplacement;
        ensemble = new ClassifierChain[aNumOfModels];
        rand = new Random(1);
    }

    public EnsembleOfClassifierChainsDT(Classifier classifier, int aNumOfModels,
            boolean doUseSamplingWithReplacement, double threshold) {
        super(classifier);
        numOfModels = aNumOfModels;
        useSamplingWithReplacement = doUseSamplingWithReplacement;
        ensemble = new ClassifierChain[aNumOfModels];
        rand = new Random(1);
        this.threshold = threshold;
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
        MultiLabelDecisionTemplate = new MLDT(ensemble, threshold);
        MultiLabelDecisionTemplate.build(trainingSet);
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception,
            InvalidDataException {

        return MultiLabelDecisionTemplate.makePredictionInternal(instance);

    }
}
