package featureLearn;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import ec.EvolutionState;
import ec.Individual;
import ec.gp.GPIndividual;
import ec.gp.GPProblem;
import ec.gp.GPTree;
import ec.simple.SimpleFitness;
import featureGrouping.MutualInformationMap;
import gp.DoubleData;
import other.DatasetUtils;
import other.Util;

import java.math.BigDecimal;
import java.util.Arrays;

public class FeatureLearnerProblem extends GPProblem {
    private static final boolean ALL_SUBSAMPLES = true;
    public static boolean normalisePostCreation;
    public static boolean scalePostCreation;
    public static boolean roundPostCreation;
    private static Cache<Long, Double> CACHED_MUTUAL_INFO = createCache(1_000_000);
    private static int hits;
    private static int misses;
    public double[] xVals;
    public int currInstIndx;
    public double[][] pcaVals;

    public static <K, V> Cache<K, V> createCache(final int maxEntries) {
        //  return new ConcurrentLinkedHashMap.Builder<K, V>().maximumWeightedCapacity(maxEntries).concurrencyLevel(Runtime.getRuntime().availableProcessors()).build();
        return Caffeine.newBuilder().maximumSize(maxEntries).build();
    }

    public double[][] getOutputsForSaving(EvolutionState state, int i, GPIndividual individual) {
        return getAllOutputs(state, i, individual);
    }
    public static void main(String[] args) {
        //Hash collision test
        int collisions = 0;
        for (int i = 0; i < 1E6; i++) {
            double[][] random = new double[10][1000];
            for (int j = 0; j < 10; j++) {
                for (int k = 0; k < 1000; k++) {
                    random[j][k] = Math.random();
                }
            }
            long i1 = TreesOutput.longHashCode(random);
            if (CACHED_MUTUAL_INFO.getIfPresent(i1) != null) {
                System.err.println("Collision");
                System.out.println(i1);
                collisions++;
            }
            if (i % 10000 == 0) System.out.println(i / 1E6d * 100);
            CACHED_MUTUAL_INFO.put(i1, 0d);
        }
        System.out.printf("Collisions: %d %.2f%%\n", collisions, (collisions / 1E6d) * 100);
    }

    public static int checkIfValidSolution(double[][] outputs) {
        int numFeatures = outputs.length;
        for (int i = 0; i < numFeatures; i++) {
            if (outputs[i] == null) {
                //Okay, so not a valid solution. fitness will be -ve numer of invalid trees.
                int count = 0;
                for (int j = i; j < numFeatures; j++) {
                    if (outputs[j] == null) {
                        count++;
                    }
                }
                return -count;
            }
        }
        return 0;
    }

    public static double[][] getSubsampleInputToUse() {
        if (FeatureLearner.NUM_SUBSAMPLES > 1 || FeatureLearner.multipleNoisyXVals[0].length < 100) {
            return FeatureLearner.subsamples[Util.randomInt(FeatureLearner.NUM_SUBSAMPLES)];
        } else {
            return FeatureLearner.multipleXVals;
        }
    }

    public double[][] getAllOutputs(EvolutionState state, int threadnum, GPIndividual gpInd) {
        if (gpInd instanceof CachedOutputIndividual) {
            double[][] outputs = ((CachedOutputIndividual) gpInd).getOutputs();
            if (outputs != null) {
                return outputs;
            }
        }
        GPTree[] trees = gpInd.trees;
        return getAllOutputs(state, threadnum, trees);

    }

    public double[][] getAllOutputs(EvolutionState state, int threadnum, GPTree[] trees) {
        double[][] outputs = new double[trees.length][];

        for (int i = 0; i < trees.length; i++) {
            outputs[i] = getOutputs(state, threadnum, new DoubleData(), trees[i], FeatureLearner.multipleNoisyXVals);
        }
        return outputs;

    }


    double[] getOutputs(EvolutionState state, int threadnum, DoubleData input, GPTree tree, double[][] multipleXVals) {
        int numSources = multipleXVals.length;
        int numInputs = multipleXVals[0].length;
        double[] results = new double[numInputs];
        for (int i = 0; i < numInputs; i++) {
            this.currInstIndx = i;
            this.xVals = new double[numSources];
            for (int j = 0; j < numSources; j++) {
                //TODO: Careful....
                this.xVals[j] = multipleXVals[j][i];
            }
            //instance-major, doh
            this.pcaVals = FeatureLearner.pcaVals[i];
            tree.child.eval(state, threadnum, input, this.stack, tree.owner, this);
            // //TODO: NaN...
            if (Double.isNaN(input.val) || Double.isInfinite(input.val)) {
                return null;
            } else {
                results[i] = input.val;
            }
        }


        if (normalisePostCreation) {
            results = DatasetUtils.normaliseFeature(results);
        } else if (scalePostCreation) {
            DatasetUtils.scaleArray(results);
        }
        if (Double.isNaN(results[0]) || Double.isInfinite(results[0])) {
            return null;
        }
        if (roundPostCreation) {
            for (int i = 0; i < results.length; i++) {
                //TODO dodgy?
                // System.out.println("rounding");
                results[i] = new BigDecimal(results[i]).setScale(5, BigDecimal.ROUND_HALF_UP).doubleValue();

            }
        }
        return results;
    }

//    private double[][] subsample(double[][] outputs, double v) {
//        if (v >= 1) {
//            return outputs;
//        }
//        if (v <= 0) {
//            return new double[0][];
//        }
//        List<Integer> collect = IntStream.range(0, outputs.length).boxed().collect(Collectors.toList());
//        Collections.shuffle(collect);
//        int toUse = (int) (outputs.length * v);
//        double[][] selected = collect.stream().limit(toUse).map(i -> outputs[i]).toArray(double[][]::new);
//        return selected;
//    }

    public void evaluate(EvolutionState state, Individual ind, int subpopulation, int threadnum) {
        if (!ind.evaluated)  // don't bother reevaluating
        {
            GPIndividual gpInd = (GPIndividual) ind;
            GPTree[] trees = gpInd.trees;

            double fitness;
            if (ALL_SUBSAMPLES) {
                fitness = 0;
                for (double[][] subsample : FeatureLearner.subsamples) {
                    double[][] outputs = getAllOutputsForEval(state, threadnum, gpInd, subsample);
                    double result = internalMeasureFitnessForInputs(subsample, outputs);
                    if (result < 0 || !Double.isFinite(result)) {
                        fitness = result;
                        break;
                    }
                    fitness += result;
                }
                if (Double.isFinite(fitness) && fitness > 0) {
                    fitness /= FeatureLearner.NUM_SUBSAMPLES;
                }
            } else {
                double[][] outputs = getAllOutputsForEval(state, threadnum, gpInd);
                fitness = internalMeasureFitness(outputs);
            }
            if (Double.isInfinite(fitness)) {
                fitness = -Double.MAX_VALUE;
            }
            SimpleFitness f = ((SimpleFitness) ind.fitness);
            f.setFitness(state, fitness, false);

            ind.evaluated = true;

        }
    }

    private double[][] getAllOutputsForEval(EvolutionState state, int threadnum, GPIndividual gpInd) {
        GPTree[] trees = gpInd.trees;
        double[][] outputs = new double[trees.length][];

        for (int i = 0; i < trees.length; i++) {
            outputs[i] = getOutputs(state, threadnum, new DoubleData(), trees[i], getSubsampleInputToUse());
        }
        return outputs;
    }


    private double[][] getAllOutputsForEval(EvolutionState state, int threadnum, GPIndividual gpInd, double[][] input) {
        GPTree[] trees = gpInd.trees;
        double[][] outputs = new double[trees.length][];

        for (int i = 0; i < trees.length; i++) {
            outputs[i] = getOutputs(state, threadnum, new DoubleData(), trees[i], input);
        }
        return outputs;
    }


    public double internalMeasureFitness(double[][] outputs) {

        int count = checkIfValidSolution(outputs);
        if (count < 0) return count;
        double mi = cachedMI(getSubsampleInputToUse(), outputs);

        return mi;// / FeatureLearner.baseMI;
    }


    public double internalMeasureFitnessForInputs(double[][] inputs, double[][] outputs) {
        //    try {
        int numFeatures = outputs.length;


        int count = checkIfValidSolution(outputs);
        if (count < 0) return count;
        double mi = cachedMI(inputs, outputs);

        return mi;// / FeatureLearner.baseMI;
    }

    Double cachedMI(double[][] multipleXVals, double[][] outputs) {
        double mi;
        TreesOutput thisOut = new TreesOutput(outputs);
        long hashcode = thisOut.longHashCode();
        return CACHED_MUTUAL_INFO.get(hashcode, a -> getMI(multipleXVals, outputs));
    }

    public double getMI(double[][] multipleXVals, double[][] outputs) {
        return MutualInformationMap.getMultiVarMutualInformationVers2(multipleXVals, outputs);
        //return MutualInformationMap.getMultiVarMutualInformation(multipleXVals, outputs, noise);
    }

    static class TreesOutput {

        private final double[][] outputs;

        public TreesOutput(double[][] outputs) {

            this.outputs = outputs;
        }

        public static long longHashCode(double[][] _2darray) {
            long result = 1;
            for (double[] element : _2darray) {
                long elementHash = longHashCode(element);
                result = 31L * result + elementHash;
            }
            return result;
        }

        private static long longHashCode(double[] a) {
            if (a == null)
                return 0;

            long result = 1;
            for (double element : a) {
                long bits = Double.doubleToLongBits(element);
                result = 31L * result + bits;
            }
            return result;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            TreesOutput that = (TreesOutput) o;
            return Arrays.deepEquals(outputs, that.outputs);
        }

        @Override
        public int hashCode() {
            return Arrays.deepHashCode(outputs);
        }

        public long longHashCode() {
            return longHashCode(outputs);
        }
    }
}
