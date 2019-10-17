package featureGrouping;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Created by lensenandr on 10/08/17.
 */
public class PearsonCorrelationMap implements VariableDependencyMap, Serializable {

    public static double getAbsolutePearsonCorrelation(double[] xVals, double[] yVals) {
        double covariance = 0;
        double xVariance = 0;
        double yVariance = 0;
        double xMean = Arrays.stream(xVals).average().getAsDouble();
        double yMean = Arrays.stream(xVals).average().getAsDouble();
        for (int i = 0; i < xVals.length; i++) {
            double xDiff = xVals[i] - xMean;
            double yDiff = yVals[i] - yMean;
            covariance += (xDiff * yDiff);
            xVariance += (xDiff * xDiff);
            yVariance += (yDiff * yDiff);
        }
        double denominator = Math.sqrt(xVariance * yVariance);
        double correlation = covariance / denominator;
        return Math.abs(correlation);

    }

}
