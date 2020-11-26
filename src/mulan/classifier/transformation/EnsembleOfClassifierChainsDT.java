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

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;

/**
 * <p>
 * Implementation of the Decition Template Ensemble of Classifier Chains(DTECC) algorithm.</p>
 * <p>
 */
public class EnsembleOfClassifierChainsDT extends EnsembleOfClassifierChains {

    private MLDT MultiLabelDecisionTemplate;

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
        super(classifier,aNumOfModels,false,doUseSamplingWithReplacement);
    }

    public EnsembleOfClassifierChainsDT(Classifier classifier, int aNumOfModels,
            boolean doUseSamplingWithReplacement, double threshold) {
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        super.buildInternal(trainingSet);
        /*Building the Decision Templates matrices*/
        MultiLabelDecisionTemplate = new MLDT(ensemble, threshold);
        MultiLabelDecisionTemplate.build(trainingSet);
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception,
            InvalidDataException {

        return MultiLabelDecisionTemplate.makePredictionInternal(instance);

    }
}
