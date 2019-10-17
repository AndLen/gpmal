package data;

import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.ml.clustering.Clusterable;
import other.DatasetUtils;
import other.Util;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Created by lensenandr on 2/03/16.
 */
public class Instance implements Clusterable, Serializable {
    public static final long serialVersionUID = 42L;
    //Allow hashcode and equals while allowing instances with the same feature values...
    public final int instanceID;
    public final double[] featureValues;
    final String classLabel;
    private final int numFeatures;


    public Instance(double[] featureValues, String classLabel, int instanceID) {
        this.featureValues = featureValues;
        this.classLabel = classLabel;
        this.instanceID = instanceID;
        this.numFeatures = featureValues.length;
    }

    public double getFeatureValue(int index) {
        return featureValues[index];
    }

    public int numFeatures() {
        return numFeatures;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Instance instance = (Instance) o;
        return instanceID == instance.instanceID;

    }

    @Override
    public int hashCode() {
        return instanceID;
    }

    /**
     * @param other
     * @param featureSubset   boolean array of whether features selected.
     * @param distanceMeasure
     * @return
     */
    public double distanceTo(Instance other, boolean[] featureSubset, Util.DistanceMeasure distanceMeasure) {
        //Euclidean distance
        return distanceMeasure.distance(this, other, featureSubset);

    }

    @Override
    public String toString() {
        return "Instance{" +
                "featureValues=" + Arrays.toString(featureValues) +
                '}';
    }

    @Override
    public Instance clone() {
        return new Instance(Arrays.copyOf(featureValues, featureValues.length), classLabel, instanceID);
    }

    public Instance scaledCopy(double[] minFeatureVals, double[] maxFeatureVals) {
        double[] scaledFeatures = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            double featureValue = getFeatureValue(i);
            scaledFeatures[i] = minFeatureVals[i] == maxFeatureVals[i] ? 0 : Util.scale(featureValue, minFeatureVals[i], maxFeatureVals[i]);
        }
        return new Instance(scaledFeatures, classLabel, instanceID);
    }

    public Instance normalisedCopy(double[] featureMeans, double[] featureStdDevs) {
        double[] normalisedFeatures = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            double featureValue = getFeatureValue(i);
            double newVal = (featureValue - featureMeans[i]) / featureStdDevs[i];
            normalisedFeatures[i] = featureStdDevs[i] == 0 ? 0 : newVal;
        }
        return new Instance(normalisedFeatures, classLabel, instanceID);
    }

    public String getClassLabel() {
        if (classLabel == null) {
            throw new IllegalStateException();
        }
        return classLabel;
    }

    @Override
    public double[] getPoint() {
        return featureValues;
    }


}
