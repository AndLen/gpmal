package featureLearn;

import ec.EvolutionState;
import ec.Individual;
import ec.Statistics;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

/**
 * Created by lensenandr on 8/07/16.
 */
public class FeatureLearnStatistics extends Statistics {


    public static void maybeToCSV(EvolutionState state, int gen) {
        if (gen == 1 || (gen != 0 && gen % 10 == 0)) {
            List<List<String>> result = FeatureLearner.gpToCSV(FeatureLearner.processedInstances, state);
            //  List<List<String>> allLines = Collections.singletonList(result.get(0));
            //   String output = FeatureLearner.formatForSaving(FeatureLearner.processedInstances, allLines);
            List<List<String>> created = Collections.singletonList(result.get(1));
            String output1 = FeatureLearner.formatForSaving(FeatureLearner.processedInstances,
                    created);
            //  LOG.println(output1);

            try {
                // FeatureLearner.writeOut(allLines, output, "gp%s%dEF-" + gen);
                FeatureLearner.writeOut(created, output1, "gp%s%dCreatedF-" + gen);

            } catch (IOException e) {
                e.printStackTrace();
            }

        }
    }

    public void postEvaluationStatistics(final EvolutionState state) {
        super.postEvaluationStatistics(state);

        // for now we just print the best fitness per subpopulation.
        Individual individual = null;  // quiets compiler complaints
        int subpopIndex = -1;
        for (int x = 0; x < state.population.subpops.length; x++) {
            for (int y = 0; y < state.population.subpops[x].individuals.length; y++) {
                if (state.population.subpops[x].individuals[y] != null) {
                    if (individual == null || state.population.subpops[x].individuals[y].fitness.betterThan(individual.fitness)) {
                        individual = state.population.subpops[x].individuals[y];
                        subpopIndex = x;
                    }
                }

            }
        }

//        else if (state.evaluator.p_problem instanceof FGNNRFLSubsetProblem) {
//            GPIndividual ind = (GPIndividual) individual;
//            FGNNRFLSubsetProblem prob = (FGNNRFLSubsetProblem) state.evaluator.p_problem;
//            double[][] outputs = prob.getAllOutputs(state, 0, ind);
//            SimpleFitness f = ((SimpleFitness) ind.fitness);
//            System.out.println(f.fitness());
//
//            f.setFitness(state, prob.doSubsetFitness(outputs, FGNNRFLSubsetProblem.getIntegers(FeatureLearner.multipleXVals[0].length)), false);
//            System.out.println(f.fitness());
//        }
        int gen = state.generation;
        maybeToCSV(state, gen);
    }


}
