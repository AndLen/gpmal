package featureLearn;

import data.Instance;
import featureGrouping.ValuedFeature;
import other.DatasetUtils;
import other.Main;
import other.Util;
import tests.featureLearn.FLNeighboursFG;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import static com.google.common.io.Files.getNameWithoutExtension;
import static featureGrouping.ValuedFeature.instancesToValuedFeatures;
import static other.DatasetUtils.*;
import static other.Main.CONFIG;
import static other.Main.SetupSystem.getRawInstances;
import static other.Util.LOG;

public class RunGPMaL {
    /**
     * @param in_args in_args[0] should contain the full path to the input file.
     *                Other parameters are all set as per GP-MaL paper.
     */
    public static void main(String in_args[]) throws IOException {
        init(in_args[0]);

        if(in_args.length > 1){
            CONFIG.put("numtrees",in_args[1]);
        }

        Util.initLogging();

        FeatureLearner.processedInstances = readAndProcessData(in_args[0]);

        List<ValuedFeature> valuedFeatures = initFL();
        LOG.printf("System set up:%n%s%n%n", CONFIG.toString());

        FeatureLearner.FLResult flResult = FeatureLearner.createEncoding(FeatureLearner.processedInstances, valuedFeatures);


        List<List<String>> allLines = new ArrayList<>();
        List<List<String>> rfLines = new ArrayList<>();
        List<List<String>> results = flResult.csv;
        allLines.add(results.get(0));
        rfLines.add(results.get(1));

        String output = FeatureLearner.formatForSaving(FeatureLearner.processedInstances, allLines);
        LOG.println(output);

        String output1 = FeatureLearner.formatForSaving(FeatureLearner.processedInstances, rfLines);
        FeatureLearner.writeOut(rfLines, output1, "gp%s%dCreatedF");

        Path resolve = LOG.DIRECTORY.resolve(LOG.customLogPath + LOG.longPrefix + "-gpmal.state");
        try (
                BufferedOutputStream fout = new BufferedOutputStream(Files.newOutputStream(resolve, StandardOpenOption.CREATE_NEW));
                ObjectOutputStream oos = new ObjectOutputStream(fout);
        ) {
            oos.writeObject(flResult.state);
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        Util.shutdownThreads();

    }

    static List<ValuedFeature> initFL() {
        int numFeatures = FeatureLearner.processedInstances.get(0).numFeatures();
        List<ValuedFeature> valuedFeatures = instancesToValuedFeatures(FeatureLearner.processedInstances, numFeatures);
        if (FeatureLearner.NUM_SOURCE_MV == -1) {
            FeatureLearner.NUM_SOURCE_MV = valuedFeatures.size();
            LOG.printf("Source features set to %d.\n", FeatureLearner.NUM_SOURCE_MV);
        }
        Main.CONFIG.put("numclasses", "" + FeatureLearner.processedInstances.stream().map(Instance::getClassLabel).distinct().count());

        FeatureLearner.FEAT_INDEX = new int[numFeatures];
        for (int j = 0; j < numFeatures; j++) {
            FeatureLearner.FEAT_INDEX[j] = j;
        }
        FeatureLearner.initSourcePrefix();
        return valuedFeatures;
    }

    public static void init(String in_arg) {
        List<String> testConfig = new FLNeighboursFG().getTestConfig();
        testConfig.add(0, "dataset=" + getNameWithoutExtension(Paths.get(in_arg).getFileName().toString()));
        String[] _args = testConfig.toArray(new String[0]);

        for (String arg : _args) {
            String[] split = arg.split("=");
            if (split.length == 2) {
                CONFIG.put(split[0].trim(), split[1].trim());
            }
        }
    }

    public static List<Instance> readAndProcessData(String in_arg) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(in_arg));
        String[] header = lines.get(0).split(",");
        List<Instance> rawInstances = getRawInstances(lines, header);
        DatasetUtils.FEATURE_MIN = CONFIG.getInt("featureMin");
        DatasetUtils.FEATURE_MAX = CONFIG.getInt("featureMax");
        DatasetUtils.FEATURE_RANGE = (FEATURE_MAX - FEATURE_MIN);
        DatasetUtils.FEATURE_RANGE2 = (FEATURE_RANGE) * (FEATURE_RANGE);

        String preprocessingType = CONFIG.getProperty("preprocessing", "scale");

        List<Instance> processedInstances;
        switch (preprocessingType) {
            case "scale":
                processedInstances = scaleInstances(rawInstances);
                break;
            case "normalise":
                processedInstances = normaliseInstances(rawInstances);
                break;
            case "none":
                processedInstances = rawInstances.stream().map(Instance::clone).collect(Collectors.toList());
                break;
            default:
                throw new IllegalArgumentException(preprocessingType);
        }
        return processedInstances;
    }
}
