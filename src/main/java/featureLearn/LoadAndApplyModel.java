package featureLearn;

import ec.EvolutionState;
import featureGrouping.ValuedFeature;
import other.Util;

import java.io.BufferedInputStream;
import java.io.ObjectInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static featureLearn.RunGPMaL.initFL;

public class LoadAndApplyModel {
    public static void main(String[] args) {
        Path model_path = Paths.get(args[0]);
        try (
                BufferedInputStream fout = new BufferedInputStream(Files.newInputStream(model_path));
                ObjectInputStream oos = new ObjectInputStream(fout);
        ) {
            EvolutionState state = (EvolutionState) oos.readObject();
            System.out.println(state);

            RunGPMaL.init(args[1]);
            Util.initLogging();

            FeatureLearner.processedInstances = RunGPMaL.readAndProcessData(args[1]);
            List<ValuedFeature> valuedFeatures = initFL();
            FeatureLearner.setupInputs(valuedFeatures);

            List<List<String>> results = FeatureLearner.gpToCSV(FeatureLearner.processedInstances, state);
            System.out.println(results);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
