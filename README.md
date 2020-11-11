# Decision Templates for Ensemble of Classifer Chains
The results from the mutli-label classification experiments that include

- source code,
- experimental results report.

## Data

All the datasets have been obtained from the multi-label data repository in http://www.uco.es/kdis/mllresources/.

## Basic instructions on how to test your own fusion scheme for Ensemble of Classifier Chains

1. Overriride  the ```makePredictionInternal(Instance instance)``` method for the ```EnsembleOfClassifierChains``` located at the ```mulan.classifier.transformation``` package,
	1.  Gather the ```MultiLabelOutput ensembleMLO``` of each classifier from the ensemble, your can use the their confidences and bipartitions:
		    ```boolean[] bip = ensembleMLO.getBipartition();
            double[] conf = ensembleMLO.getConfidences();```
	1.  Combine their outputs using the choosen fusion scheme.
1. Check out the examples provided at the ```mulan.examples``` package on how to run your experiments using a [Evaluator](http://mulan.sourceforge.net/doc/mulan/evaluation/Evaluator.html "Evaluator").