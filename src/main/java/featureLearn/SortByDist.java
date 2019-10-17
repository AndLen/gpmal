package featureLearn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SortByDist {
    private double[][] instanceMajor;
    private double[][] instanceDistances;
    private List<List<Integer>> sortedNeighbours;

    public SortByDist(double[][] instanceMajor) {
        this.instanceMajor = instanceMajor;
    }

    public static double[][] findPairwiseDistances(double[][] instanceMajor) {
        int numI = instanceMajor.length;
        double[][] instanceDistances = new double[numI][numI];
        for (int i = 0; i < numI; i++) {
            instanceDistances[i][i] = -1;
            for (int j = i + 1; j < numI; j++) {
                double datDist = NearestNeighbourFLProblem.euclideanDistance(instanceMajor[i], instanceMajor[j]);
                //Yay symmetry
                instanceDistances[i][j] = datDist;
                instanceDistances[j][i] = datDist;
            }
        }
        return instanceDistances;
    }

    public double[][] getInstanceDistances() {
        return instanceDistances;
    }

    public List<List<Integer>> getSortedNeighbours() {
        return sortedNeighbours;
    }

    public SortByDist invoke() {
        int numI = instanceMajor.length;
        instanceDistances = findPairwiseDistances(instanceMajor);
        sortedNeighbours = new ArrayList<>();
        for (int i = 0; i < numI; i++) {
            //All neighbours except itself
            Integer[] indicies = new Integer[numI - 1];
            for (int j = 0; j < i; j++) {
                indicies[j] = j;
            }
            for (int j = i + 1; j < numI; j++) {
                indicies[j - 1] = j;
            }
            Arrays.sort(indicies, new FuzzyGaussianNNRankingFLProblem.AllNeighboursSorted.DistanceComparator(instanceDistances[i]));
            sortedNeighbours.add(Arrays.asList(indicies));
        }
        return this;
    }
}
