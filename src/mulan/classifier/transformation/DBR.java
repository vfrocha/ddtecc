/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mulan.classifier.transformation;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author lmello, vrocha
 */
public class DBR extends TransformationBasedMultiLabelLearner {

    private Add[] addsattr;
    private BinaryRelevance br;
    private FilteredClassifier[] singleClassifiers;

    public DBR(Classifier classifier) {
        super(classifier);
        br = new BinaryRelevance(classifier);
    }

    private void buildInternalClassifier(Instances dataSet, int i) throws Exception {
        int numAttrs = dataSet.numAttributes();
        FilteredClassifier fc = singleClassifiers[i];
        fc.setClassifier(AbstractClassifier.makeCopy(baseClassifier));

        int[] indicesToRemove = new int[numLabels];
        for (int j = 0; j < numLabels; j++) {
            if (j == i) {
                indicesToRemove[j] = numAttrs - numLabels + i;
            } else {
                indicesToRemove[j] = labelIndices[j];
            }
        }

        Remove remove = new Remove();
        remove.setAttributeIndicesArray(indicesToRemove);
        remove.setInputFormat(dataSet);
        remove.setInvertSelection(false);

        fc.setFilter(remove);
        dataSet.setClassIndex(labelIndices[i]);
        fc.buildClassifier(dataSet);
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        br.build(trainingSet);
        Instances dataSet = trainingSet.getDataSet();
        singleClassifiers = new FilteredClassifier[this.numLabels];
        Instances newdataset = expandData(dataSet);

        for (int i = 0; i < singleClassifiers.length; i++) {
            singleClassifiers[i] = new FilteredClassifier();
            buildInternalClassifier(newdataset, i);
        }

    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {
        MultiLabelOutput mout1 = br.makePrediction(instance);
        Instance transformedInstance = TransformInstance(instance,mout1);
        double[] probabilities = new double[this.numLabels];
        double threshold = 0.5;
        
        for (int i = 0; i< singleClassifiers.length; i++) {
            FilteredClassifier fc = singleClassifiers[i];
            probabilities[i] = fc.distributionForInstance(transformedInstance)[0];
        }
        
        return new MultiLabelOutput(probabilities, threshold);
    }

    private Instance TransformInstance(Instance inst, MultiLabelOutput mlo) {
        for (int j = 0; j < numLabels; j++) {
            addsattr[j].input(inst);
            inst = addsattr[j].output();
        }

        boolean[] bipart = mlo.getBipartition();
        for (int j = 0; j < numLabels; j++) {
            final int attr_index = inst.numAttributes() - numLabels + j;

            inst.setValue(attr_index, bipart[j] ? 1 : 0);
        }
        return inst;
    }

    private Instances expandData(Instances trainData) throws Exception {
        addsattr = new Add[numLabels];
        Instances newtrainData = trainData;
        for (int j = 0; j < numLabels; j++) {
            addsattr[j] = new Add();
            addsattr[j].setOptions(new String[]{"-T", "NUM"});

            addsattr[j].setAttributeIndex("last");
            addsattr[j].setAttributeName("labelAttr" + j);
            addsattr[j].setInputFormat(newtrainData);

            newtrainData = Filter.useFilter(newtrainData, addsattr[j]);
        }
        for (int i = 0; i < newtrainData.numInstances(); i++) {
            Instance inst = newtrainData.instance(i);
            for (int j = 0; j < numLabels; j++) {
                inst.setValue(j + trainData.numAttributes(),
                        inst.value(labelIndices[j]));
            }
        }
        return newtrainData;
    }

}
