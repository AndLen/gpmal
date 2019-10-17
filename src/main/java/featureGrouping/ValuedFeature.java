package featureGrouping;

import data.Instance;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by lensenandr on 9/08/17.
 */
public class ValuedFeature {
    public final double[] values;
    public final int featureID;


    public static List<ValuedFeature> instancesToValuedFeatures(List<Instance> processedInstances, int numFeatures) {
        double[][] featureValues = new double[numFeatures][processedInstances.size()];
        for (int i = 0; i < processedInstances.size(); i++) {
            Instance instance = processedInstances.get(i);
            final double[] instanceVals = instance.featureValues;
            for (int j = 0; j < instanceVals.length; j++) {
                featureValues[j][i] = instanceVals[j];
            }
        }
        List<ValuedFeature> valuedFeatures = new ArrayList<>();
        for (int i = 0; i < featureValues.length; i++) {
            valuedFeatures.add(new ValuedFeature(featureValues[i], i));
        }


        return valuedFeatures;
    }


    public ValuedFeature(double[] values, int featureID) {
        this.values = values;
        this.featureID = featureID;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ValuedFeature that = (ValuedFeature) o;

        return featureID == that.featureID;

    }

    @Override
    public int hashCode() {
        return featureID;
    }
}
