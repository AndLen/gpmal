package featureLearn;

import ec.EvolutionState;
import ec.Individual;
import ec.gp.GPIndividual;
import ec.gp.GPTree;
import ec.simple.SimpleFitness;
import net.sf.javaml.core.kdtree.KDTree;
import other.FitnessCache;
import other.Util;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.lang.Math.sqrt;


public class NearestNeighbourFLProblem extends FeatureLearnerProblem {
    public static final int numNeighbours = 10;
    public static List<List<Integer>> highDimNNs;
    public static List<List<Double>> highDimScaledDists;
    private static FitnessCache CACHED_FITNESS = new FitnessCache();

    public static double weightNthContribution(double n) {
        //n in [1,2,3..,10].
        return 1 / n;//Math.sqrt(n);
    }

    public static double euclideanDistance(double[] i1, double[] i2) {
        double sum = 0;
        for (int i = 0; i < i1.length; i++) {
            double diff = i1[i] - i2[i];
            sum += (diff * diff);
        }
        return sqrt(sum);
    }

    public static List<List<Double>> getDists(double[][] instanceMajor, List<List<Integer>> highDimNNs, int numNeighbours, boolean scale) {
        if (numNeighbours < 0) numNeighbours = highDimNNs.get(0).size();
        List<List<Double>> highDimScaledDists = new ArrayList<>();
        for (int i = 0; i < highDimNNs.size(); i++) {
            List<Integer> theseNeighbours = highDimNNs.get(i);
            List<Double> dists = new ArrayList<>(numNeighbours);
            for (int j = 0; j < numNeighbours; j++) {
                double dist = euclideanDistance(instanceMajor[i], instanceMajor[theseNeighbours.get(j)]);
                dists.add(dist);
            }
            highDimScaledDists.add(dists);
        }

        //Scale them proportional to the median.
        if (scale) {
            scaleByMedian(numNeighbours, highDimScaledDists);
        }
        return highDimScaledDists;
    }

    public static void scaleByMedian(int numNeighbours, List<List<Double>> distances) {
        double[] allDists = distances.stream().flatMapToDouble(l -> l.stream().mapToDouble(d -> d)).toArray();
        Arrays.sort(allDists);
        double minDist = allDists[0];
        double medianDist = allDists[allDists.length / 2];
        double maxDist = allDists[allDists.length - 1];

        for (int i = 0; i < distances.size(); i++) {
            List<Double> highDimScaledDist = distances.get(i);
            for (int j = 0; j < numNeighbours; j++) {
                double v = highDimScaledDist.get(j);
                //In range [1,median,max].
                v = Util.scale(v, minDist, medianDist) + 1;
                highDimScaledDist.set(j, v);
            }
        }
    }

    public double internalMeasureFitness(double[][] outputs) {

        int count = checkIfValidSolution(outputs);
        if (count < 0) return count;
        synchronized (NearestNeighbourFLProblem.class) {
//            if (neighboursMap == null) {
//                neighboursMap = initNeighbours(Util.transposeMatrix(FeatureLearner.multipleXVals));
//            }
            if (highDimNNs == null) {
                double[][] instanceMajor = Util.transposeMatrix(FeatureLearner.multipleXVals);
                highDimNNs = new JavaMLNNF().neighbours(instanceMajor, numNeighbours);
                highDimScaledDists = getDists(instanceMajor, highDimNNs, numNeighbours, true);
            }
        }
        boolean[] valids = DecisionBoundaryFLProblem.lookForNoise(outputs);
        int numValid = (int) Util.booleanStream(valids).filter(Boolean::booleanValue).count();
        double[][] validOutputs;
        if (numValid == outputs.length) {
            validOutputs = outputs;
        } else {
            validOutputs = new double[numValid][outputs[0].length];
            int nextI = 0;
            for (int i = 0; i < outputs.length; i++) {
                if (valids[i]) {
                    System.arraycopy(outputs[i], 0, validOutputs[nextI++], 0, outputs[0].length);
                }
            }
        }
        if (validOutputs.length == 0) {
            // =/
            return -Double.MAX_VALUE;
        }
        return cachedFitness(validOutputs, this::measureNeighbourPreservationKDDist);

    }

    private double measureNeighbourPreservationKDDist(double[][] outputs) {
        double[][] instanceMajor = Util.transposeMatrix(outputs);

        List<List<Integer>> lowDimNNs = new JavaMLNNF().neighbours(instanceMajor, numNeighbours);
        if (lowDimNNs.size() == 0) {
            //Not enough unique neighbours!!
            return -Double.MAX_VALUE;
        }
        //compare them
        double sumScores = 0;
        double numAgrees = 0;
        double numDisagrees = 0;
        for (int i = 0; i < instanceMajor.length; i++) {
            List<Integer> high = highDimNNs.get(i);
            List<Integer> low = lowDimNNs.get(i);
            List<Double> thisHighDim = highDimScaledDists.get(i);
            for (int j = 0; j < high.size(); j++) {
                Integer index = high.get(j);
                //   if (low.indexOf(index) == j) {
                if (low.contains(index)) {
                    numAgrees++;
                    //nearer neighbours in dist = more impt.
                    sumScores += 1 / thisHighDim.get(j);
                } else {
                    numDisagrees++;
                }
            }
        }
        //1/1 + 1/2 + ... + 1/10 rougly is 2.92 -> 3
        sumScores /= (instanceMajor.length * numNeighbours);
        return sumScores;

    }

    //    private double measureNeighbourPreservation(double[][] outputs) {
//        double[][] instanceMajor = Util.transposeMatrix(outputs);
//        ConcurrentMap<Integer, List<Integer>> map = initNeighbours(instanceMajor, numNeighbours);
//        //compare them
//        double numAgrees = 0;
//        double numDisagrees = 0;
//        for (int i = 0; i < instanceMajor.length; i++) {
//            List<Integer> highDim = neighboursMap.get(i);
//            List<Integer> lowDim = map.get(i);
//            for (int index : highDim) {
//                if (lowDim.contains(index)) {
//                    numAgrees++;
//                } else {
//                    numDisagrees++;
//                }
//            }
//        }
//        return numAgrees / (numAgrees + numDisagrees);
//
//    }

    public double cachedFitness(double[][] outputs, ComputeFitness computeFitness) {
        TreesOutput thisOut = new TreesOutput(outputs);
        long hashcode = thisOut.longHashCode();
        return CACHED_FITNESS.getFitness(hashcode, computeFitness, outputs);

    }

    @Override
    public void evaluate(EvolutionState state, Individual ind, int subpopulation, int threadnum) {
        if (!ind.evaluated)  // don't bother reevaluating
        {
            try {
                GPIndividual gpInd = (GPIndividual) ind;
                GPTree[] trees = gpInd.trees;

                double fitness;

                double[][] outputs = getAllOutputs(state, threadnum, gpInd);
                fitness = internalMeasureFitness(outputs);
//            if (Double.isInfinite(fitness)) {
//                fitness = -Double.MAX_VALUE;
//            }
                SimpleFitness f = ((SimpleFitness) ind.fitness);
                boolean isOptimal = fitness == 1;

//            if (Double.isFinite(fitness)) {
//                isOptimal = fitness == 1 / (outputs[0].length / (Math.pow(2, trees.length)));
//
//            }
                f.setFitness(state, fitness, isOptimal);

                ind.evaluated = true;

            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                if (!ind.evaluated) {
                    System.err.println("Uh oh");
                }
            }
        }
    }

    public interface NearestNeighbourFinder {
    }

    public static class JavaMLNNF implements NearestNeighbourFinder {

        private List<List<Integer>> neighbours(double[][] instanceMajor, int numNeighbours) {

            instanceMajor = Util.deepCopy2DArray(instanceMajor);
            KDTree tree = new KDTree(instanceMajor[0].length);
            for (Integer i = 0; i < instanceMajor.length; i++) {
                tree.insert(instanceMajor[i], i);
            }

            try {
                List<List<Integer>> neighbours = new ArrayList<>();
                for (double[] anInstanceMajor : instanceMajor) {
                    Object[] nearest = tree.nearest(anInstanceMajor, numNeighbours + 1);
                    //Don't count itself...
                    List<Integer> nearest_int = IntStream.range(1, nearest.length).mapToObj(i -> (Integer) nearest[i]).collect(Collectors.toList());
                    // List<Integer> nearest_int = Arrays.stream(nearest).map(v -> (Integer) v).collect(Collectors.toList());
                    neighbours.add(nearest_int);
                }
                return neighbours;
            } catch (Exception e) {
                //Means fewer unique neighbours than needed...
                return Collections.emptyList();
            }
        }
    }
}
