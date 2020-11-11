/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mulan.experiments;

import java.util.ArrayList;
import java.util.List;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.classifier.transformation.EnsembleOfClassifierChainsDT;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.AverageMAE;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.AverageRMSE;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ErrorSetSize;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.ExampleBasedSpecificity;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.LogLoss;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.MacroPrecision;
import mulan.evaluation.measure.MacroRecall;
import mulan.evaluation.measure.MacroSpecificity;
import mulan.evaluation.measure.MeanAveragePrecision;
import mulan.evaluation.measure.MeanSquaredError;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroAUC;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.MicroSpecificity;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
import mulan.evaluation.measure.RootMeanSquaredError;
import mulan.evaluation.measure.SubsetAccuracy;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Utils;

public class DTECCExp {

    public static void main(String[] args) throws Exception {
        String arffFilename = Utils.getOption("arff", args); // e.g. -arff emotions.arff
        String xmlFilename = Utils.getOption("xml", args); // e.g. -xml emotions.xml

        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);

        int ensembleSize = 50;
        
        EnsembleOfClassifierChainsDT learnerDT =  new EnsembleOfClassifierChainsDT(new NaiveBayes(), ensembleSize, true);
        EnsembleOfClassifierChains learnerMV =  new EnsembleOfClassifierChains(new NaiveBayes(), ensembleSize, false, true);
        EnsembleOfClassifierChains learnerME =  new EnsembleOfClassifierChains(new NaiveBayes(), ensembleSize, true, true);
        
        Evaluator eval = new Evaluator();
        MultipleEvaluation results;

        int someFolds = 10;
        int numLabels = dataset.getNumLabels();
        
        List<Measure> measures = new ArrayList<>(27);
        measures.add(new ExampleBasedAccuracy());
        measures.add(new ExampleBasedPrecision());
        measures.add(new ExampleBasedRecall());
        measures.add(new HammingLoss());
        measures.add(new SubsetAccuracy());
        measures.add(new ExampleBasedFMeasure());

        measures.add(new AverageMAE(numLabels));
        measures.add(new AveragePrecision());
        measures.add(new AverageRMSE(numLabels));
        measures.add(new Coverage());
        measures.add(new ErrorSetSize());
        measures.add(new ExampleBasedSpecificity());
        measures.add(new LogLoss());
        measures.add(new MacroFMeasure(numLabels));
        measures.add(new MacroPrecision(numLabels));
        measures.add(new MacroRecall(numLabels));
        measures.add(new MacroSpecificity(numLabels));
        measures.add(new MeanAveragePrecision(numLabels));
        measures.add(new MeanSquaredError());
        measures.add(new MicroAUC(numLabels));
        measures.add(new MicroFMeasure(numLabels));
        measures.add(new MicroPrecision(numLabels));
        measures.add(new MicroRecall(numLabels));
        measures.add(new MicroSpecificity(numLabels));
        measures.add(new OneError());
        measures.add(new RankingLoss());
        measures.add(new RootMeanSquaredError());
        
        System.out.println("arffFilename;Learner;ExampleBasedAccuracy;ExampleBasedPrecision;ExampleBasedRecall;HammingLoss;SubsetAccuracy;ExampleBasedFMeasure;AverageMAE;AveragePrecision;AverageRMSE;Coverage;ErrorSetSize;ExampleBasedSpecificity;LogLoss;MacroFMeasure;MacroPrecision;MacroRecall;MacroSpecificity;MeanAveragePrecision;MeanSquaredError;MicroAUC;MicroFMeasure;MicroPrecision;MicroRecall;MicroSpecificity;OneError;RankingLoss;RootMeanSquaredError;");
        
        results = eval.crossValidate(learnerMV, dataset, measures, someFolds);
        System.out.println(arffFilename + ';' + learnerMV.getClass().getSimpleName() + "MV;" + results.toCSV());
        results = eval.crossValidate(learnerME, dataset, measures, someFolds);
        System.out.println(arffFilename + ';' + learnerME.getClass().getSimpleName() + "ME;" + results.toCSV());
        results = eval.crossValidate(learnerDT, dataset, measures, someFolds);
        System.out.println(arffFilename + ';' + learnerDT.getClass().getSimpleName() + "(S);" + results.toCSV());
    }
}
