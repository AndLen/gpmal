package featureLearn;

import ec.EvolutionState;
import ec.Individual;
import ec.gp.GPIndividual;
import ec.gp.GPTree;
import ec.simple.SimpleFitness;
import featureGrouping.PearsonCorrelationMap;
import other.Util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class DecisionBoundaryFLProblem extends FeatureLearnerProblem {

    public static final double SIMILARITY_THRESHOLD = 0.95;

    public static DBResult getDBResult(double[][] outputs) {
        int numCF = outputs.length;
        int numI = outputs[0].length;

        List<Integer> noiseDBs = new ArrayList<>();

        boolean[][] membershipMatrix = new boolean[numCF][numI];
        int[] encodedMemberships = new int[numI];
        //  double sumMargins = 0;
        double[][] dbBounds = new double[numCF][2];
        for (int i = 0; i < numCF; i++) {
            double[] thisCF = outputs[i];
//            double[] dbBound = findDecisionBoundary(thisCF);
//            if (dbBound[0] == dbBound[1]) {
//                noiseDBs.add(i);
//                // return null;
//            }
            double[] copy = Arrays.copyOf(thisCF, thisCF.length);
            Arrays.sort(copy);
            for (int j = 0; j < copy.length; j++) {
                if (copy[i] > 0) {
                    dbBounds[i][0] = copy[i - 1];
                    dbBounds[i][1] = copy[i];
                }
            }
            //dbBounds[i] = dbBound;
            double midBoundary = 0;
            //(dbBound[0] + dbBound[1]) / 2;
            //   sumMargins += (dbBounds[1] - dbBounds[0]);
            for (int j = 0; j < thisCF.length; j++) {
                if (thisCF[j] > midBoundary) {
                    //Give it a "1" in the encoding.
                    //Using a IP-address style approach -- 2^n
                    membershipMatrix[i][j] = true;
                    encodedMemberships[j] += (int) Math.pow(2, i);
                }
            }
        }
        //System.out.println(membershipCounts);
        int[] membershipCounts = new int[(int) Math.pow(2, numCF)];
        for (int membership : encodedMemberships) {
            membershipCounts[membership]++;
        }

        return new DBResult(membershipMatrix, membershipCounts);
    }

    /**
     * Surprisingly complicated...
     *
     * @param outputs
     * @return
     */
    public static double[][] removeDupsAndNoise(double[][] outputs) {
        int numOriginally = outputs.length;
        boolean[] validOutputs = lookForNoise(outputs);
        boolean[][] dupMappings = new boolean[numOriginally][numOriginally];


        //Lets see which dups we have...
        for (int i = 0; i < validOutputs.length; i++) {
            //Only check dups for still valid (non-noisy) ones.
            if (validOutputs[i]) {
                for (int j = i + 1; j < validOutputs.length; j++) {
                    if (validOutputs[j]) {
                        double corr = PearsonCorrelationMap.getAbsolutePearsonCorrelation(outputs[i], outputs[j]);
                        if (corr >= SIMILARITY_THRESHOLD) {
                            //Symmetric
                            dupMappings[i][j] = true;
                            dupMappings[j][i] = true;
                        }
                    }

                }
            }
        }
        int maxDupIndex = getMaxDupIndex(dupMappings);
        //Go through and remove the one that is "most redundant" with all others at each step until there's no redundant
        //features remaining.
        while (maxDupIndex >= 0) {
            //Make this one invalid
            validOutputs[maxDupIndex] = false;
            for (int i = 0; i < numOriginally; i++) {
                //Remove the dups for this index as it no longer is being used.
                dupMappings[maxDupIndex][i] = false;
                //And remove the dup for this index in the other's as it has been removed.
                dupMappings[i][maxDupIndex] = false;
            }

            maxDupIndex = getMaxDupIndex(dupMappings);
        }

        int numValid = (int) Util.booleanStream(validOutputs).filter(Boolean::booleanValue).count();
        //If we didn't remove any, shortcut. Otherwise, new array with the remaining valid indicies.
        if (numValid == numOriginally) {
            return outputs;
        } else {
            double[][] newOutputs = new double[numValid][];
            int nxIdx = 0;
            for (int i = 0; i < validOutputs.length; i++) {
                if (validOutputs[i]) {
                    newOutputs[nxIdx++] = outputs[i];
                }

            }
            return newOutputs;
        }
    }

    public static boolean[] lookForNoise(double[][] outputs) {
        int numOriginally = outputs.length;
        boolean[] validOutputs = new boolean[numOriginally];
        //Assume all true to start with
        Arrays.fill(validOutputs, true);

        //Remove noise first, easier.
        for (int i = 0; i < numOriginally; i++) {
            double[] output = outputs[i];
            double[] copy = Arrays.copyOf(output, output.length);
            Arrays.sort(copy);
            //We round to 5dp, so anything less than this? or too sensitive anyway....
            if (copy[copy.length - 1] - copy[0] < 0.0001) {
                validOutputs[i] = false;
            }
        }
        return validOutputs;
    }

    private static int getMaxDupIndex(boolean[][] dupMappings) {
        int num = dupMappings.length;
        int maxDups = 0;
        int maxDupIndex = -1;
        for (int i = 0; i < num; i++) {
            int numDups = 0;
            for (int j = 0; j < num; j++) {
                if (dupMappings[i][j]) {
                    numDups++;
                }
            }
            if (numDups > maxDups) {
                maxDups = numDups;
                maxDupIndex = i;
            }
        }
        return maxDupIndex;
    }

    public static double calcJointEntropy(DBResult res, int totalInstances) {
        double entropySum = 0;
        int[] membershipCounts = res.membershipCounts;
        for (int i = 0; i < membershipCounts.length; i++) {
            int mC = membershipCounts[i];
            if (mC > 0) {
                double jointProb = mC / (double) (totalInstances);
                entropySum += /*mC */ (jointProb * Util.log2(jointProb));
            }
        }
        return -1 * entropySum;
    }


    private double measureTrimmedFitness(double[][] outputs) {

        //   int numTrees = outputs.length;
        DBResult dbR = getDBResult(outputs);
        //   System.out.println(Arrays.toString(membershipCounts));
        //  int numInstances = outputs[0].length;
        //int optimalMaxCount = (int) Math.ceil(numInstances / Math.pow(2, numTrees));

        //  double maxCount = Arrays.stream(dbR.membershipCounts).max().getAsInt();
//        double minCount = Arrays.stream(dbR.membershipCounts).min().getAsInt();
//        int[] membershipCounts = dbR.membershipCounts;
//        double geometricMeanCount = getSpecialGeometricMean(membershipCounts);
//        double harmonicMeanCount = getSpecialHarmonicMean(membershipCounts);
//        double medianCount = Util.getMedian(membershipCounts);
        //    int[] nonZeros = Arrays.stream(dbR.membershipCounts).filter(i -> i != 0).toArray();
        //   double numNonZeros = nonZeros.length;
//        double numZeros = dbR.membershipCounts.length - numNonZeros;
//        Covariance covariance = new Covariance();
//        double maxMemSame = 0;
//        double sumMemSame = 0;
//        double maxCovar = 0;
//        double maxMemCovar = 0;
//        double sumCovar = 0;
//        double sumMemCovar = 0;
//        double weightedImpurity = 0;
//        for (int i = 0; i < numTrees; i++) {
//            double numPos = Util.booleanStream(dbR.membershipMatrix[i]).filter(b -> b.equals(Boolean.TRUE)).count();
//            double treePurity = Math.max(numPos, numInstances - numPos) / (double) numInstances;
//            double treeImpurity = 1 - treePurity;
//            double thisCovarSum = 0;
//            for (int j = 0; j < numTrees; j++) {
//                if (i != j && !(dbR.noiseDBs.contains(i) || dbR.noiseDBs.contains(j))) {
//                    double absCovar = Math.abs(covariance.covariance(outputs[i], outputs[j]));
//                    double absMemCovar = Math.abs(membershipCovariance(dbR.membershipMatrix[i], dbR.membershipMatrix[j]));
//                    maxCovar = Math.max(maxCovar, absCovar);
//                    maxMemCovar = Math.max(maxMemCovar, absMemCovar);
//                    sumCovar += absCovar;
//                    sumMemCovar += absMemCovar;
//                    thisCovarSum += absMemCovar;
//
//
//                }
//            }
//            for (int j = i + 1; j < numTrees; j++) {
//                int numMatches = numMatches(dbR.membershipMatrix[i], dbR.membershipMatrix[j]);
//                maxMemSame = Math.max(maxMemSame, numMatches);
//                sumMemSame += numMatches;
//            }
//            thisCovarSum /= (numTrees - 1);
//            weightedImpurity += (treeImpurity * thisCovarSum);
//        }
//        sumCovar /= ((numTrees * numTrees));

        //     double var = Util.getVariance(dbR.membershipCounts);
        //    double nonZeroVar = Util.getVariance(nonZeros);
//
//        double numTinyBins = Arrays.stream(nonZeros).filter(i -> i < 10d).count();
//        //return numValidTrees * (1 / (1 + maxMemSame));
//
//        int[] mcCopy = Arrays.copyOf(membershipCounts, membershipCounts.length);
//        Arrays.sort(mcCopy);
//
//        double lowerQuartile = mcCopy[mcCopy.length / 4];

//        MutualInformationCalculatorDiscrete miCalc = null;
//        try {
//            miCalc = new MutualInformationCalculatorDiscrete(2,2);
//
//        } catch (Exception e) {
//            return -Double.MAX_VALUE;
        //       }
        //first is "time", second is feature index.

//        int[][] forCalc = new int[numTrees][numInstances];
//
//        for (int i = 0; i < numTrees; i++) {
//            for (int j = 0; j < numInstances; j++) {
//                forCalc[i][j] = dbR.membershipMatrix[i][j] ? 1 : 0;
//            }
//        }

        //first is "time", second is feature index.

//        int[][] forCalc = new int[numInstances][numTrees];
//        double[][] forCalc2 = new double[numInstances][numTrees];
//        for (int i = 0; i < numTrees; i++) {
//            for (int j = 0; j < numInstances; j++) {
//                forCalc[j][i] = dbR.membershipMatrix[i][j] ? 1 : 0;
//                forCalc2[j][i] = outputs[i][j];
//            }
//        }
//        MultiInformationCalculatorDiscrete miCalc = new MultiInformationCalculatorDiscrete(2, numTrees);
//        //  MultiInfoCalculatorKraskov2 miCalcCont = new MultiInfoCalculatorKraskov2();
//
//        miCalc.initialise();
//        miCalc.addObservations(forCalc);
        //  miCalcCont.initialise(numTrees);
//        try {
//            miCalcCont.setObservations(forCalc2);
//            double mi = miCalcCont.computeAverageLocalOfObservations();
//            return mi / numTrees;
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new Error(e);
//        }
//        double sumMI = 0;
//        double maxMI = 0;
//        double minMI = Double.MAX_VALUE;
//        for (int i = 0; i < numTrees; i++) {
//            for (int j = i + 1; j < numTrees; j++) {
//                miCalc.initialise();
//                miCalc.addObservations(forCalc[i], forCalc[j]);
//                double mi = miCalc.computeAverageLocalOfObservations();
//                sumMI += mi;
//                maxMI = Math.max(maxMI, mi);
//                minMI = Math.min(minMI, mi);
//            }
        //       }

        //return (2-mi) * (1 / (double) numTrees);
        //return (1/(mi+2)) *
        // return (1 / (maxCount)) * (numZeros + 1);
        return fitty(dbR, outputs);
        //double jointEntropy = //calcJointEntropy(outputs);
        //calcJointEntropy(dbR, numInstances);
        // double mi = miCalc.computeAverageLocalOfObservations();
        //between 0< and 1. Minimise.
        //double maxCountRatio = maxCount / (double) numInstances;
        //maxCountRatio/=10;
        //double meanEntropy = getEntropyStats(dbR);
        //return (1+jointEntropy)*(1+meanEntropy);//(Math.log(1+nonZeroVar))*1/(maxCountRatio);//

        // *Math.log(numTrees)*-1*(Math.log(maxCountRatio));//(-1*Util.log2(maxCountRatio));//*(maxCountRatio);//*Math.log(10+numNonZeros);//*(1/Math.sqrt(maxCount));//Math.log(numTrees+1);
        //return (1 / maxCount) * (2 - mi);
//        return (2 - mi) * numTrees;// * (1/Math.sqrt(numTrees));
        // return (2-mi) * (1 / maxCount);
        //return (1/maxCount)*(maxMI);
        //return  -miCalc.computeAverageLocalOfObservations();
        //return  (1/miCalc.computeAverageLocalOfObservations())*(1/maxCount);
        //return (1 / miCalc.computeAverageLocalOfObservations()) * (1 / (1 + numIdenticalTrees + dbR.noiseDBs.size()));
//        return miCalc.computeAverageLocalOfObservations()/(1+numIdenticalTrees+dbR.noiseDBs.size());
        //Disgusting, but..
        //return lowerQuartile == 0 ? -(dbR.membershipCounts.length-numNonZeros) / (1 + maxMemSame) : lowerQuartile / (1 + maxMemSame);
//        return ((double)(mcCopy[mcCopy.length/4])) / (1 + maxMemSame);

//        return (numTrees - (dbR.noiseDBs.size() + numIdenticalTrees)) * weightedImpurity;
        //return (1 / (numTinyBins + 1)) * (1 / (maxMemCovar + 1));


        //   return (harmonicMeanCount) * (1 / (1 + maxMemCovar));


        //  double meanLimitedSeperation = Arrays.stream(dbR.dbBounds).mapToDouble(b -> Math.min(b[1] - b[0], 1))
        //          .average().getAsDouble();
//        if (maxCount <= 50) {
//            System.out.println(Arrays.toString(membershipCounts));
//        }
        //Minimise variance!
        //return (nonZeroVar == 0 ? Double.MAX_VALUE : 1 / (nonZeroVar// * (1 - (nonZeros.length-dbR.membershipCounts.length))
        //));// * (1 / (double)maxCount);
        //Encourage the median to increase
        //  return Math.sqrt(nonZeros.length)/highestCovariance;
        //return (optimalMaxCount / (double) maxCount)
        // return (1/ maxCount) / (1 + maxMemCovar);
        //return Math.sqrt(medianCount)/Math.sqrt(1+highestCovariance);
        //    ;// /Math.pow(nonZeros.length,1/4d);
        //        + (1 / (double) numInstances) * meanLimitedSeperation;
        //return (optimalMaxCount / (double) maxCount) + 0.001 * (medianCount == 0 ? 0 : (optimalMaxCount / medianCount));
        //     return (optimalMaxCount / (double) maxCount) + 0.001 * ((optimalMaxCount / var));//+0.01 * (sumMargins / numCF);


        //DECENT?
        //return numValidTrees * (1 / (1 + maxMemSame));
        //return (1 / maxCount) * (1 / (1 + maxMemCovar));
        //return (1/nonZeroVar) / (1 + maxMemCovar);
        //return (1/maxCount)*Math.log(1.01+minCount) * (1 / (1 + maxMemCovar));

    }

    public double fitty(DBResult dbR, double[][] outputs) {
        return calcJointEntropy(dbR, outputs[0].length);
    }

    public double[][] getOutputsForSaving(EvolutionState state, int i, GPIndividual individual) {
        return removeDupsAndNoise(getAllOutputs(state, i, individual));
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
                boolean isOptimal = false;

//            if (Double.isFinite(fitness)) {
//                isOptimal = fitness == 1 / (outputs[0].length / (Math.pow(2, trees.length)));
//
//            }
                f.setFitness(state, fitness, false);

                ind.evaluated = true;

            } finally {
                if (!ind.evaluated) {
                    System.err.println("Uh oh");
                }
            }
        }
    }

    public double internalMeasureFitness(double[][] outputs) {
        int count = FeatureLearnerProblem.checkIfValidSolution(outputs);
        if (count < 0) return count;
        double[][] trimmedOutputs = removeDupsAndNoise(outputs);
        return measureTrimmedFitness(trimmedOutputs);

    }

    public static class DBResult {
        public final int[] membershipCounts;
        private final boolean[][] membershipMatrix;

        public DBResult(boolean[][] membershipMatrix, int[] membershipCounts) {
            this.membershipMatrix = membershipMatrix;
            this.membershipCounts = membershipCounts;
        }
    }
}