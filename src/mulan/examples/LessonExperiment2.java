package mulan.examples;

import java.io.File;
import java.io.FileWriter;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;


public class LessonExperiment2 {

    public static void main(String[] args) {
        try {
            String path = "/home/vrocha/Documents/mulan-1.5.0/mulan/data/"; 
            String filestem = "emotions"; 
            String percentage = "50";  

            System.out.println("Loading the dataset");
            MultiLabelInstances mlDataSet = new MultiLabelInstances(path + filestem + ".arff", path + filestem + ".xml");

            // split the data set into train and test
            Instances dataSet = mlDataSet.getDataSet();
            RemovePercentage rmvp = new RemovePercentage();
            rmvp.setInvertSelection(true);
            rmvp.setPercentage(Double.parseDouble(percentage));
            rmvp.setInputFormat(dataSet);
            Instances trainDataSet = Filter.useFilter(dataSet, rmvp);

            rmvp = new RemovePercentage();
            rmvp.setPercentage(Double.parseDouble(percentage));
            rmvp.setInputFormat(dataSet);
            Instances testDataSet = Filter.useFilter(dataSet, rmvp);

            MultiLabelInstances train = new MultiLabelInstances(trainDataSet, path + filestem + ".xml");
            MultiLabelInstances test = new MultiLabelInstances(testDataSet, path + filestem + ".xml");

            Evaluator eval = new Evaluator();
            Evaluation results;

            Classifier brClassifier = new NaiveBayes();
            BinaryRelevance br = new BinaryRelevance(brClassifier);
            br.setDebug(true);
            br.build(train);
            results = eval.evaluate(br, test, train);
            FileWriter fw = new FileWriter(new File("/tmp/results"));
            fw.write(results.toString());
            fw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}