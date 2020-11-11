package mulan.classifier.transformation;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Multi-label Decision Templates.
 *
 * @author vfrocha
 */
public class MLDT extends TransformationBasedMultiLabelLearner {

    private static final long serialVersionUID = -446512578928821388L;
    private MultiLabelLearner[] classifiers;
    private int classifiersLength = 10;
    private MLDTBR[] DT;
    private double confThreshold = 0.5;
    private boolean useMedian = false;

    public static final SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

    double mstime(long t) {
        return (System.nanoTime() - t) / 1e6;
    }

    /**
     * Creates a new instance
     *
     * @param classifiers the base-level classification algorithm that will be
     * used for training each of the binary models
     * @param aChain
     */
    public MLDT(MultiLabelLearner[] classifiers, double confThreshold) {
        super();
        this.classifiers = classifiers;
        this.classifiersLength = classifiers.length;
        this.confThreshold = confThreshold;
    }

    /**
     * Creates a new instance
     *
     * @param classifiers the base-level classification algorithm that will be
     * used for training each of the binary models
     * @param aChain
     */
    public MLDT(MultiLabelLearner[] classifiers) {
        super();
        this.classifiers = classifiers;
        this.classifiersLength = classifiers.length;
    }

    private void buildTemplates(MultiLabelInstances mlinstances) throws Exception {
        DT = new MLDTBR[numLabels];
        for (int i = 0; i < numLabels; i++) {
            DT[i] = new MLDTBR();
        }

        Instances newtrainData = mlinstances.getDataSet();
        for (int d = 0; d < newtrainData.size(); d++) {
            Instance inst = newtrainData.get(d);
            boolean[] trueLabels = getTrueLabels(inst, numLabels, mlinstances.getLabelIndices());

            for (int j = 0; j < classifiersLength; j++) {
                double[] confidencesForInstance = classifiers[j].makePrediction(inst).getConfidences();

                for (int i = 0; i < numLabels; i++) {
                    DT[i].addColumn(j, confidencesForInstance, trueLabels[i]);
                }
            }

        }

        for (int i = 0; i < numLabels; i++) {
            DT[i].computeTemplates();
        }
    }

    private boolean[] getTrueLabels(Instance instance, int numLabels, int[] labelIndices) {

        boolean[] trueLabels = new boolean[numLabels];
        for (int counter = 0; counter < numLabels; counter++) {
            int classIdx = labelIndices[counter];
            String classValue = instance.attribute(classIdx).value((int) instance.value(classIdx));
            trueLabels[counter] = classValue.equals("1");
        }

        return trueLabels;
    }

    @Override
    protected void buildInternal(MultiLabelInstances all) throws Exception {
        buildTemplates(all);
    }

    private double[] bipartition2double(boolean[] bipartition) {
        double[] confidences = new double[numLabels];

        for (int i = 0; i < numLabels; i++) {
            if (bipartition[i]) {
                confidences[i] = 1.0;
            } else {
                confidences[i] = 0.0;
            }
        }
        return confidences;
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        double[][] confidencesMatrix = new double[classifiersLength][numLabels];
        boolean[] bipartition = new boolean[numLabels];

        for (int c = 0; c < classifiersLength; c++) {
            confidencesMatrix[c] = classifiers[c].makePrediction(instance).getConfidences();
        }

        for (int i = 0; i < numLabels; i++) {
            bipartition[i] = DT[i].makePredictionIndividual(i, confidencesMatrix);
        }

        return new MultiLabelOutput(bipartition);
    }

    public MultiLabelOutput makePrediction(double[][] confidencesMatrix) {
        boolean[] bipartition = new boolean[numLabels];

        for (int i = 0; i < numLabels; i++) {
            bipartition[i] = DT[i].makePredictionIndividual(i, confidencesMatrix);
        }

        return new MultiLabelOutput(bipartition);
    }

    /**
     * Multilabel Decision Template builder using the Binary Relevance approach
     *
     */
    private class MLDTBR {

        private int posivitesCount = 0;
        private int negativesCount = 0;
        private final double[][] positiveMatrix = new double[classifiersLength][numLabels];
        private final double[][] negativeMatrix = new double[classifiersLength][numLabels];
        private final ArrayList<double[]> positiveInstances = new ArrayList<>();
        private final ArrayList<double[]> nagetiveInstances = new ArrayList<>();
        private boolean[] relevancyMask;

        public MLDTBR() {
        }

        public void addColumn(int index, double[] confidences, boolean trueLabel) {
            if (trueLabel) {
                //Add to positiveMatrix
                addMatrixMean(index, positiveMatrix, confidences);
                addInstances(positiveInstances, confidences);
                posivitesCount++;
            } else {
                //Add to negativeMatrix
                addMatrixMean(index, negativeMatrix, confidences);
                addInstances(nagetiveInstances, confidences);
                negativesCount++;
            }
        }

        public void computeTemplates() {
            posivitesCount = posivitesCount / classifiersLength;
            negativesCount = negativesCount / classifiersLength;

            //compute positive matrix mean
            if (useMedian) {
                computeMatrixMedian(positiveMatrix, positiveInstances);
                computeMatrixMedian(negativeMatrix, nagetiveInstances);

            } else {
                computeMatrixMean(positiveMatrix, posivitesCount);
                computeMatrixMean(negativeMatrix, negativesCount);
            }

        }

        private void computeMatrixMean(double[][] matrix, int count) {
            for (int i = 0; i < classifiersLength; i++) {
                for (int j = 0; j < numLabels; j++) {
                    matrix[i][j] /= count;
                }
            }
        }

        private void computeMatrixMedian(double[][] matrix, ArrayList<double[]> instances) {
            int len = classifiersLength;

            for (int j = 0; j < len; j++) {
                for (int l = 0; l < numLabels; l++) {
                    ArrayList<Double> allValuesL = new ArrayList<>();
                    for (int i = j; i < instances.size(); i += len) {
                        allValuesL.add(instances.get(i)[l]);
                    }
                    double median = median(allValuesL);
                    matrix[j][l] = median;
                }
            }
        }

        private double median(ArrayList<Double> numArray) {

            Collections.sort(numArray);
            int middle = numArray.size() / 2;
            double medianValue = 0;
            if (numArray.size() % 2 == 1) {
                medianValue = numArray.get(middle);
            } else {
                medianValue = (numArray.get(middle - 1) + numArray.get(middle)) / 2;
            }

            return medianValue;
        }

        private void addMatrixMean(int index, double[][] matrix, double[] confidences) {
            for (int j = 0; (j < numLabels) && (j < confidences.length); j++) {
                matrix[index][j] += confidences[j];
            }
        }

        private void addInstances(ArrayList<double[]> instances, double[] confidences) {
            instances.add(confidences);
        }

        public boolean makePrediction(double[][] confidencesMatrix) {
            double euclidPositiveDist = 0;
            double euclidNegativeDist = 0;

            for (int i = 0; i < classifiersLength; i++) {
                for (int j = 0; (j < numLabels) && (j < confidencesMatrix[i].length); j++) {
                    euclidPositiveDist += Math.abs(confidencesMatrix[i][j] - positiveMatrix[i][j]);
                    euclidNegativeDist += Math.abs(confidencesMatrix[i][j] - negativeMatrix[i][j]);
                }
            }

            double conf = 1 - (euclidPositiveDist / (euclidPositiveDist + euclidNegativeDist));

            return conf > confThreshold;
        }

        //INDIVIDUAL PREDICTION TYPE
        public boolean makePredictionIndividual(int index, double[][] confidencesMatrix) {
            double euclidPositiveDist = 0;
            double euclidNegativeDist = 0;

            for (int i = 0; i < classifiersLength; i++) {
                euclidPositiveDist += Math.abs(confidencesMatrix[i][index] - positiveMatrix[i][index]);
                euclidNegativeDist += Math.abs(confidencesMatrix[i][index] - negativeMatrix[i][index]);
            }
            double conf = 1 - (euclidPositiveDist / (euclidPositiveDist + euclidNegativeDist));

            return conf > confThreshold;
        }

    }

    /*Debug only*/
    //c1->[d1,d2,...,dl]
    //c2->[d1,d2,...,dl]
    //c3->[d1,d2,...,dl]
    private void printMatrix(double[][] confidencesMatrix) {
        for (int i = 0; i < classifiersLength; i++) {
            System.out.println(Arrays.toString(confidencesMatrix[i]));
        }
        System.out.println("");
    }

}
