package featureLearn;

//

import org.apache.commons.math3.distribution.NormalDistribution;
import other.Util;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;


public class FuzzyGaussianNNRankingFLProblem extends NearestNeighbourFLProblem {
    public static int numSDsForPenalty;// = 20;
    public static double norm;
    public static double[] gaussianPenalties;
    public static int sdSizeForPenalty = 20;

    public static double[] initPenalties(int numSDsForPenalty, double sdSizeForPenalty) {
        double[] gaussianPenalties = new double[numSDsForPenalty];
        NormalDistribution nD = new NormalDistribution(0, sdSizeForPenalty);
        for (int i = 0; i < numSDsForPenalty; i++) {
            int bounds = (i + 1);
            gaussianPenalties[i] = //(1/(double)(i+1));//
                    nD.probability(-bounds, +bounds);
        }
        System.out.println(Arrays.toString(gaussianPenalties));
        return gaussianPenalties;
    }

    public static double weightDiscrepancy(int actual, int found) {
        int absDiff = Math.abs(actual - found);
        if (absDiff == 0) {
            return 1;
        } else if (absDiff > gaussianPenalties.length) {
            return 0;
        } else {
            //      System.out.printf("wanted: %d got: %d diff: %d penalty: %.2f", actual, found, absDiff, 1-gaussianPenalties[absDiff - 1]);
            return 1 - gaussianPenalties[absDiff - 1];
        }

    }

    public double internalMeasureFitness(double[][] outputs) {

        int count = checkIfValidSolution(outputs);
        if (count < 0) return count;
        synchronized (NearestNeighbourFLProblem.class) {

            if (highDimNNs == null) {
                numSDsForPenalty = outputs[0].length;
                gaussianPenalties = initPenalties(numSDsForPenalty, sdSizeForPenalty);
                double[][] instanceMajor = Util.transposeMatrix(FeatureLearner.multipleXVals);
                highDimNNs = new AllNeighboursSorted().neighbours(instanceMajor, -1);
                norm = IntStream.range(0, instanceMajor.length - 1).mapToDouble(i -> 1 / (double) (i + 1)).sum();

            }
        }

        return cachedFitness(outputs, this::fuzzyGaussianRankingFitness);

    }

    double fuzzyGaussianRankingFitness(double[][] outputs) {
        double[][] instanceMajor = Util.transposeMatrix(outputs);

        List<List<Integer>> lowDimNNs = new AllNeighboursSorted().neighbours(instanceMajor, -1);

        //compare them
        double sumScores = 0;
        for (int i = 0; i < instanceMajor.length; i++) {
            List<Integer> high = highDimNNs.get(i);
            List<Integer> low = lowDimNNs.get(i);
            for (int highI = 0; highI < high.size(); highI++) {
                Integer neighbour = high.get(highI);
                int lowI = low.indexOf(neighbour);
                //How impt it is times how off it is
                sumScores += (1 / (double) (highI + 1)) * weightDiscrepancy(highI, lowI);
            }
        }
        //If perfectly preserved all, then n(n-1) "1s".
        sumScores /= (instanceMajor.length * norm);
        return sumScores;
    }

    public static class AllNeighboursSorted implements NearestNeighbourFinder {
        public List<List<Integer>> neighbours(double[][] instanceMajor, int numNeighbours) {
            SortByDist sortByDist = new SortByDist(instanceMajor).invoke();
            List<List<Integer>> sortedNeighbours = sortByDist.getSortedNeighbours();
            return sortedNeighbours;

        }

        public static class DistanceComparator implements Comparator<Integer> {

            private final double[] distances;

            public DistanceComparator(double[] distances) {
                this.distances = distances;
            }

            @Override
            public int compare(Integer o1, Integer o2) {
                return Double.compare(distances[o1], distances[o2]);
            }
        }

    }

}
