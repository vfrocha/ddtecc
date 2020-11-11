package mulan.examples;

import mulan.classifier.MultiLabelOutput;

public class LessonExperiment4 {

    public static void main(String[] args) {
        double[] probabilities = new double[]{0.57,0.87,0.1};
        boolean[] bipartition = new boolean[]{true,false,false};
        double threshold = 0.6;
        MultiLabelOutput multiLabelOutput = new MultiLabelOutput(bipartition);
        MultiLabelOutput multiLabelOutput1 = new MultiLabelOutput(probabilities);
        MultiLabelOutput multiLabelOutput2 = new MultiLabelOutput(probabilities, threshold);

        System.out.println(multiLabelOutput);
        System.out.println(multiLabelOutput1);
        System.out.println(multiLabelOutput2);
    }
}