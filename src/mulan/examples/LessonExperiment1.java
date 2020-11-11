package mulan.examples;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.DBR;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;


public class LessonExperiment1 {


    public static void main(String[] args) {
        try {
            String path = "/home/vrocha/Documents/mulan-1.5.0/mulan/data/"; 
            String filestem = "emotions"; 

            System.out.println("Loading the dataset");
            MultiLabelInstances mlDataSet = new MultiLabelInstances(path + filestem + ".arff", path + filestem + ".xml");

            Instances dataSet = mlDataSet.getDataSet();
            Instance remove = dataSet.remove(0);
            
            MultiLabelInstances train = new MultiLabelInstances(dataSet, path + filestem + ".xml");
            
            Classifier brClassifier = new NaiveBayes();
            DBR br = new DBR(brClassifier);
            
            br.build(train);
            
            MultiLabelOutput prediction = br.makePrediction(remove);
            System.out.println(prediction);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}