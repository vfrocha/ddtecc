
# Decision Templates for Ensemble of Classifer Chains
The results from the mutli-label classification experiments that include

- source code,
- experimental results report.

## Data

All the datasets have been obtained from the multi-label data repository in http://www.uco.es/kdis/mllresources/.

## Reproducing the experiments

### Requirements


| Mulan  |  Weka  | Java |
|  ----- | -------| -----|
|  1.5.0 | 3.7.10 | 1.7  |


### Instructions

1. Download Mulan 1.5 from http://mulan.sourceforge.net/download.html,
2. Download the full datasets used in the experiments in its Mulan version (ARFF and XML file),
3. Download the DTECC source files from the [src folder](https://github.com/vfrocha/dtecc/tree/main/src/mulan "src folder") in this repositirory;
4. Place the downloaded files in their respective packages:
	- [EnsembleOfClassifierChainsDT.java](https://github.com/vfrocha/dtecc/blob/main/src/mulan/classifier/transformation/EnsembleOfClassifierChainsDT.java "EnsembleOfClassifierChainsDT.java") and [MLDT.java](https://github.com/vfrocha/dtecc/blob/main/src/mulan/classifier/transformation/MLDT.java "MLDT.java") on mulan.classifier.transformation;
	- [DTECCExp.java](https://github.com/vfrocha/dtecc/blob/main/src/mulan/experiments/DTECCExp.java "DTECCExp.java") on mulan.experiments;
5. Build and run the project. For example, if the source file of the experiment is in the same directory with emotions.arff, emotions.xml, weka.jar and mulan.jar of the distribution package, to run this experiment on Windows, you can type the following command:
```javac -cp mulan.jar;weka.jar DTECCExp.java java -cp mulan.jar;weka.jar;. DTECCExp -arff emotions.arff -xml emotions.xml```	

## Basic instructions on how to test your own fusion scheme for Ensemble of Classifier Chains

1. Replace the ```makePredictionInternal(Instance instance)``` method for the ```EnsembleOfClassifierChains``` located at the ```mulan.classifier.transformation``` package,
	1.  Gather the ```MultiLabelOutput ensembleMLO``` of each classifier from the ensemble, your can use the their confidences and bipartitions in the fusion process:
		    ```boolean[] bip = ensembleMLO.getBipartition();
            double[] conf = ensembleMLO.getConfidences();```
	1.  Combine their outputs using the choosen fusion scheme.
1. Check out the examples provided at the ```mulan.examples``` package on how to run your experiments using a [Evaluator](http://mulan.sourceforge.net/doc/mulan/evaluation/Evaluator.html "Evaluator").
