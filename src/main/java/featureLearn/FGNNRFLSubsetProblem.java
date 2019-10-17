package featureLearn;

import other.Util;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class FGNNRFLSubsetProblem extends FuzzyGaussianNNRankingFLProblem {
    public static final int BASE = 2;
    public static final double UNIFORM_RATIO = 0.1;
    //The set of high dimensional neighbours we care about in low dim.
    static List<Integer> rankingsToRespect;
    static List<List<Integer>> highDimRespectedInstances;

    public static List<Integer> getLogStyleRankings(int numInstances) {
        //10 + 20 + 40 + 80...
        //[0,10], [11,30], [31, 70], [71, 150],...
        //Need a multiple of 10 for now.
        List<Integer> toRespect = new ArrayList<>();

        for (int i = 0; ; i++) {
            int base;
            if (i == 0) {
                base = 0;
            } else {
                //i = 1 gives base 10, i = 2 gives base 30..
                base = 10 * ((int) Math.pow(BASE, i) - 1);
            }
            //Each pool has 10.
            int stepMultiplier = (int) Math.pow(BASE, i);
            for (int j = 0; j < 10; j++) {
                //If i = 1, then will go [20
                int nextF = base + (stepMultiplier * j);
                if (nextF >= numInstances) {
                    System.out.println(toRespect);
                    return toRespect;
                }
                toRespect.add(nextF);
            }
        }
    }

    public static double weightAccordingly() {
        return 1;// / (double) (i + 1);
    }

    public static void initHighDimNNs(double[][] outputs) {
        numSDsForPenalty = outputs[0].length;
        gaussianPenalties = initPenalties(numSDsForPenalty, sdSizeForPenalty);
        rankingsToRespect = //IntStream.range(0, 20).boxed().collect(Collectors.toList());//
                getLogStyleRankings(outputs[0].length);
        // getUniformRankings(outputs[0].length);

        Util.LOG.printf("Respected rankings: %s\n", rankingsToRespect);
        double[][] instanceMajor = Util.transposeMatrix(FeatureLearner.multipleXVals);
        highDimNNs = new AllNeighboursSorted().neighbours(instanceMajor, -1);
        highDimRespectedInstances = getRespectedInstances(instanceMajor, rankingsToRespect, highDimNNs);
        norm = rankingsToRespect.stream().mapToDouble(i -> weightAccordingly()).sum();


    }

    public static List<List<Integer>> getRespectedInstances(double[][] instanceMajor, List<Integer> respectedRankings, List<List<Integer>> allNNs) {
        List<List<Integer>> respectedInstances = new ArrayList<>();

        for (int i = 0; i < instanceMajor.length; i++) {
            List<Integer> myRespectedNeighbours = new ArrayList<>();
            List<Integer> myHighDimNNs = allNNs.get(i);
            for (int j = 0; j < respectedRankings.size(); j++) {
                myRespectedNeighbours.add(myHighDimNNs.get(respectedRankings.get(j)));
            }
            respectedInstances.add(myRespectedNeighbours);
        }
        return respectedInstances;
    }

    public static List<Integer> getNNRankings(int length) {
        return IntStream.range(0, length).boxed().collect(Collectors.toList());
    }

    public static List<Integer> getUniformRankings(int length) {
        return getUniformRankings(0, length);
    }

    public static List<Integer> getUniformRankings(int start, int finished) {
        int length = finished - start;
        int numRankings = (int) Math.ceil(length * UNIFORM_RATIO);
        int spacing = length / numRankings;
        List<Integer> toRespect = new ArrayList<>(numRankings);
        for (int i = 0; i < length; i += spacing) {
            toRespect.add(start + i);
        }
        return toRespect;
    }

    public static double doSubsetFitness(double[][] instanceMajor, List<Integer> rankingsToRespect, List<List<Integer>> highDimRespectedInstances, double norm) {
        List<List<Integer>> lowDimNNs = new AllNeighboursSortedSomeRanks(rankingsToRespect, highDimRespectedInstances).neighbours(instanceMajor, -1);
        if (lowDimNNs.size() == 0) {
            return -Double.MAX_VALUE;
        }
        //compare them
        double sumScores = 0;
        try {
            for (int i = 0; i < instanceMajor.length; i++) {
                //Get the correct neighbour from high dim space.
                List<Integer> highRespectedRankings = highDimRespectedInstances.get(i);
                List<Integer> lowRespectedRankings = lowDimNNs.get(i);
                for (int highI = 0; highI < highRespectedRankings.size(); highI++) {
                    Integer neighbour = highRespectedRankings.get(highI);
                    int lowI = lowRespectedRankings.indexOf(neighbour);
                    //How impt it is times how off it is
                    sumScores += (weightAccordingly()) * weightDiscrepancy(highI, lowI);
                }
                //  System.nanoTime();
            }
            //If perfectly preserved all, then n(n-1) "1s".
            sumScores /= (instanceMajor.length * norm);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return sumScores;
    }

    double fuzzyGaussianRankingFitness(double[][] outputs) {
        double[][] instanceMajor = Util.transposeMatrix(outputs);
        return cachedFitness(instanceMajor, iM ->
                doSubsetFitness(iM, rankingsToRespect, highDimRespectedInstances, norm));
        //doSubsetFitness(instanceMajor, respectedRankings);
    }

    @Override
    public double internalMeasureFitness(double[][] outputs) {

        int count = checkIfValidSolution(outputs);
        if (count < 0) return count;
        synchronized (NearestNeighbourFLProblem.class) {

            if (highDimNNs == null) {
                initHighDimNNs(outputs);
            }
        }

        return fuzzyGaussianRankingFitness(outputs);

    }

    public static class AllNeighboursSortedSomeRanks extends AllNeighboursSorted {
        public final List<Map<Integer, Double>> rawDistances = new ArrayList<>();
        public final List<List<IndexedDistance>> sortedDistances = new ArrayList<>();
        private List<Integer> respectedRankings;
        private List<List<Integer>> respectedInstances;

        public AllNeighboursSortedSomeRanks(List<Integer> respectedRankings, List<List<Integer>> respectedInstances) {
            this.respectedRankings = respectedRankings;
            this.respectedInstances = respectedInstances;


        }

        public List<List<Integer>> neighbours(double[][] instanceMajor, int numNeighbours) {
            int numI = instanceMajor.length;
            numNeighbours = respectedRankings.size();
            List<List<Integer>> sortedNeighbours = new ArrayList<>();
            Map<Integer, Double> idToRawDist = new HashMap<>();
            for (int i = 0; i < numI; i++) {
                List<Integer> myHighNNs = respectedInstances.get(i);
                List<IndexedDistance> myIDs = new ArrayList<>(numNeighbours);
                sortedDistances.add(myIDs);
                /*
                 * We have to purposefully NOT add the neighbours in order, otherwise it'll figure out that if it just
                 * outputs a constant value then everything will be already ordered, as all distances will be 0 in the
                 * low dim space...
                 */
                for (int j = numNeighbours - 1; j >= 0; j--) {
                    int nextNN = myHighNNs.get(j);
                    double datDist = euclideanDistance(instanceMajor[i], instanceMajor[nextNN]);
                    idToRawDist.put(nextNN, datDist);
                    myIDs.add(new IndexedDistance(nextNN, datDist));
                }

                //Don't need to do this if we reverse the loop order
                //Collections.shuffle(myIDs);
                Collections.sort(myIDs);

                rawDistances.add(idToRawDist);
                List<Integer> sortedIndicies = new ArrayList<>(numNeighbours);
                for (IndexedDistance myID : myIDs) {
                    sortedIndicies.add(myID.neighbourIndex);
                }
                sortedNeighbours.add(sortedIndicies);
            }

            return sortedNeighbours;

        }
    }


    public static class IndexedDistance implements Comparable<IndexedDistance> {
        public final int neighbourIndex;
        public final double dist;

        public IndexedDistance(int neighbourIndex, double dist) {

            this.neighbourIndex = neighbourIndex;
            this.dist = dist;
        }

        @Override
        public int compareTo(IndexedDistance o) {
            return Double.compare(this.dist, o.dist);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            IndexedDistance that = (IndexedDistance) o;
            return neighbourIndex == that.neighbourIndex;
        }

        @Override
        public int hashCode() {
            return neighbourIndex;
        }
    }
}
