package featureLearn;

import data.Instance;
import ec.EvolutionState;
import ec.Evolve;
import ec.Statistics;
import ec.gp.GPIndividual;
import ec.simple.SimpleStatistics;
import ec.util.Parameter;
import ec.util.ParameterDatabase;
import featureCreate.gp.FeatureCreatorStatistics;
import featureGrouping.ValuedFeature;
import other.Main;
import other.Util;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.PrincipalComponents;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static featureGrouping.ValuedFeature.instancesToValuedFeatures;
import static other.Main.CONFIG;
import static other.Util.IN_A_JAR;
import static other.Util.LOG;

public class FeatureLearner {

    private static final boolean DO_ALL = true;
    private static final boolean ADD_NOISE = false;
    private static final double EPSILON = 0.001;
    private static final double MIN_NOISE = EPSILON * 0.001;
    public static int NUM_VAR_PCA = 5;
    public static int NUM_COMPONENTS = 3;
    public static String SOURCE_PREFIX;
    public static int[] FEAT_INDEX;// = 1;
    public static double[] xVals;// = {.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0};
    public static double[][] multipleXVals;
    public static double[][] multipleNoisyXVals;
    public static double[][][] pcaVals;
    public static String[][] pcaFeatNames;
    public static double[][][] subsamples;
    public static int NUM_SUBSAMPLES = 1;
    public static double baseMI;
    public static double baseMultiInfo;
    public static boolean MULTIVARIATE = true;
    public static int NUM_SOURCE_MV = -1;
    static List<Instance> processedInstances;
    private static Path OUT_DIR;

    public static void main(String[] args) throws IOException {
        Main.SetupSystem system = new Main.SetupSystem(args).invoke();
        processedInstances = system.getProcessedInstances();

        int numFeatures = system.getNumFeatures();
        List<ValuedFeature> valuedFeatures = instancesToValuedFeatures(processedInstances, numFeatures);

        if (NUM_SOURCE_MV == -1) {
            //NUM_SOURCE_MV = Math.max(2, Math.min(valuedFeatures.size() / 4, 5));
            NUM_SOURCE_MV = valuedFeatures.size();
            LOG.printf("Source features set to %d.\n", NUM_SOURCE_MV);
        }
        Main.CONFIG.put("numclasses", "" + processedInstances.stream().map(Instance::getClassLabel).distinct().count());
        if (DO_ALL) {

            List<List<String>> allLines = new ArrayList<>();
            List<List<String>> rfLines = new ArrayList<>();
            int numSources = MULTIVARIATE ? NUM_SOURCE_MV : 1;
            for (int i = 0; (i + numSources) <= numFeatures; i += numSources) {
                FEAT_INDEX = new int[numSources];
                for (int j = 0; j < numSources; j++) {
                    FEAT_INDEX[j] = i + j;
                }
                initSourcePrefix();
                List<List<String>> results = createEncoding(processedInstances, valuedFeatures).csv;
                allLines.add(results.get(0));
                rfLines.add(results.get(1));

            }
            String output = formatForSaving(processedInstances, allLines);
            LOG.println(output);

            //writeOut(allLines, output, "gp%s%dEF");
            String output1 = formatForSaving(processedInstances, rfLines);
            writeOut(rfLines, output1, "gp%s%dCreatedF");
        } else {
            FEAT_INDEX = new int[]{0};
            initSourcePrefix();
            createEncoding(processedInstances, valuedFeatures);

        }

        Util.shutdownThreads();
    }

    public static void writeOut(List<List<String>> allLines, String output, String fileNameFormat) throws IOException {
        int numLines = allLines.get(0).size();
        writeOut(numLines, output, fileNameFormat);
    }

    public static void writeOut(int numLines, String output, String fileNameFormat) throws IOException {
        String dataset = CONFIG.getProperty("dataset").replaceAll("/", "");
        if (OUT_DIR == null) {
            //So it all ends up in the same dir...
            OUT_DIR = Util.LOG.DIRECTORY.resolve(String.format("%s-%s-%s/", dataset, new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss").format(new Date()), Util.LOG.timestamp));
            Files.createDirectories(OUT_DIR);
        }
        String outPrefix = String.format(fileNameFormat, dataset, numLines);
        String outCSV = outPrefix + ".csv";
        Path csvPath = OUT_DIR.resolve(outCSV);
        Files.write(csvPath, Collections.singletonList(output));

//        Map<String, String> env = new HashMap<>();
//        env.put("create", "true");
//        // locate file system by using the syntax
//        // defined in java.net.JarURLConnection
//        Path zipPath = OUT_DIR.resolve(outPrefix + ".zip");
//        URI zipURI = URI.create("jar:file:" + zipPath.toAbsolutePath().toString());
//        try (FileSystem zipfs = FileSystems.newFileSystem(zipURI, env)) {
//            // copy a file into the zip file
//            Files.write(zipfs.getPath(csvPath.getFileName().toString()), Collections.singletonList(output));
//        }
    }

    public static String formatForSaving(List<Instance> processedInstances, List<List<String>> allLines) {
        StringBuilder sb = new StringBuilder();
        for (int j = 0; j < allLines.size(); j++) {
            sb.append(allLines.get(j).get(0)).append(", ");
        }
        sb.append("class\n");

        int numRFs = allLines.get(0).size();
        for (int i = 1; i < numRFs; i++) {
            for (int j = 0; j < allLines.size(); j++) {
                sb.append(allLines.get(j).get(i)).append(", ");
            }
            sb.append(processedInstances.get(i - 1).getClassLabel());
            sb.append("\n");
        }
        return sb.toString();
    }

    public static void initSourcePrefix() {
        StringBuilder sourcePrefix = new StringBuilder();
        for (int i = 0; i < FEAT_INDEX.length && i < 10; i++) {
            int fI = FEAT_INDEX[i];
            sourcePrefix.append(fI).append("-");
            if (i == 9) {
                sourcePrefix.append("x-");
            }
        }
        //Remove final '-'
        sourcePrefix.deleteCharAt(sourcePrefix.length() - 1);
        SOURCE_PREFIX = sourcePrefix.toString();
    }

    static FLResult createEncoding(List<Instance> processedInstances, List<ValuedFeature> valuedFeatures) {
        setupInputs(valuedFeatures);
        int numFeatures = FEAT_INDEX.length;

        int numInstances = multipleNoisyXVals[0].length;

        if (CONFIG.containsKey("subsamples")) {
            String subsamples = CONFIG.getProperty("subsamples");
            if (subsamples.equals("lim30")) {
                NUM_SUBSAMPLES = Math.min(10, numInstances / 30);
            } else {
                NUM_SUBSAMPLES = CONFIG.getInt("subsamples");
            }
            LOG.printf("Using %d subsamples with %d instances in each\n", NUM_SUBSAMPLES, numInstances / NUM_SUBSAMPLES);
        }
        if (NUM_SUBSAMPLES > 1) {
            subsamples = new double[NUM_SUBSAMPLES][][];
            List<Integer> shuffled = IntStream.range(0, numInstances).boxed().collect(Collectors.toList());
            Collections.shuffle(shuffled, new Random(300260289));
            int instPerSamp = numInstances / NUM_SUBSAMPLES;
            for (int sampIx = 0; sampIx < NUM_SUBSAMPLES; sampIx++) {
                subsamples[sampIx] = new double[numFeatures][];
                for (int featIx = 0; featIx < numFeatures; featIx++) {
                    subsamples[sampIx][featIx] = new double[instPerSamp];
                    for (int instIx = 0; instIx < instPerSamp; instIx++) {
                        subsamples[sampIx][featIx][instIx] = multipleNoisyXVals[featIx][shuffled.get(sampIx * instPerSamp + instIx)];
                    }
                }
            }
        } else {
            subsamples = new double[][][]{multipleNoisyXVals};
        }


        EvolutionState state = doGP();
        return new FLResult(gpToCSV(processedInstances, state), state);

    }

    public static void setupInputs(List<ValuedFeature> valuedFeatures) {
        Random random = new Random(0);

        int numFeatures = FEAT_INDEX.length;

        multipleXVals = new double[numFeatures][];
        multipleNoisyXVals = new double[numFeatures][];
        //Instance major..
        //pcaVals = new [valuedFeatures.get(0).values.length];
        for (int fI = 0; fI < numFeatures; fI++) {
            int index = FEAT_INDEX[fI];
            ValuedFeature feature = valuedFeatures.get(index);

            double[] values = Arrays.copyOf(feature.values, feature.values.length);
            multipleXVals[fI] = values;


            if (ADD_NOISE) {
                multipleNoisyXVals[fI] = new double[multipleXVals[fI].length];
                for (int j = 0; j < multipleXVals[fI].length; j++) {
                    //Better results with duplicate values (e.g. RW), removes 0s (div by 0 less likely )
                    double noise = MIN_NOISE + (random.nextDouble() * (EPSILON - MIN_NOISE));
                    multipleNoisyXVals[fI][j] = multipleXVals[fI][j] + noise;
                }

                //  LOG.println(Arrays.toString(multipleXVals[fI]));
                //  LOG.println(Arrays.toString(multipleNoisyXVals[fI]));
            } else {
                multipleNoisyXVals[fI] = new double[multipleXVals[fI].length];
                for (int j = 0; j < multipleXVals[fI].length; j++) {
                    multipleNoisyXVals[fI][j] = multipleXVals[fI][j] + EPSILON;
                }
            }

            // LOG.println(feature.featureID + " : " + Arrays.toString(values));
        }
        initPCASelection();


        LOG.println(numFeatures + " source features being used: " + Arrays.toString(FEAT_INDEX));
    }

    public static void initPCASelection() {
        double[][] instanceMajor = Util.transposeMatrix(multipleXVals);
        double[][] instanceMajorCopy = Util.transposeMatrix(multipleXVals);
        int numInstances = instanceMajor.length;
        try {
            ArrayList<Attribute> attributes = (ArrayList<Attribute>) Arrays.stream(FEAT_INDEX)
                    .mapToObj(v -> new Attribute("f" + v)).collect(Collectors.toList());
            Instances raw = new Instances("Instances", attributes, numInstances);
            raw.setClassIndex(-1);
            for (double[] i : instanceMajorCopy) {
                raw.add(new DenseInstance(1, i));
            }

            PrincipalComponents principalComponents = new PrincipalComponents();
            principalComponents.setMaximumAttributes(NUM_COMPONENTS);
            principalComponents.setInputFormat(raw);
            Instances pcaed = Filter.useFilter(raw, principalComponents);
            int numPC = pcaed.numAttributes();

            NUM_COMPONENTS = numPC;
            pcaVals = new double[numInstances][NUM_COMPONENTS][];
            pcaFeatNames = new String[NUM_COMPONENTS][];
            for (int i = 0; i < numPC; i++) {
                Attribute attribute = pcaed.attribute(i);
                System.out.println(attribute);
                Pattern compile = Pattern.compile("f\\d*");
                Matcher matcher = compile.matcher(attribute.name());
                List<String> orderedVariables = new ArrayList<>();
                while (matcher.find()) {
                    orderedVariables.add(matcher.group());
                }
                //Just in case..
                int numSubvars = Math.min(NUM_VAR_PCA, orderedVariables.size());
                orderedVariables = orderedVariables.subList(0, numSubvars);
                pcaFeatNames[i] = orderedVariables.toArray(new String[0]);

                for (int j = 0; j < numInstances; j++) {
                    pcaVals[j][i] = new double[numSubvars];
                    for (int k = 0; k < numSubvars; k++) {
                        int featIndex = Integer.parseInt(orderedVariables.get(k).substring(1));
                        //System.out.println(featIndex);
                        pcaVals[j][i][k] = instanceMajor[j][featIndex];
                    }
                }
            }
            System.out.println(Arrays.deepToString(pcaFeatNames));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    //@NotNull
    public static List<List<String>> gpToCSV(List<Instance> processedInstances, EvolutionState state) {

        Statistics statistics = state.statistics;
        GPIndividual individual;
        FeatureLearnerProblem problem = (FeatureLearnerProblem) state.evaluator.p_problem;

        individual = (GPIndividual) ((SimpleStatistics) statistics).best_of_run[0];

        double[][] out = problem.getOutputsForSaving(state, 0, individual);

        List<List<String>> result = formatAsCSV(processedInstances, out, FEAT_INDEX, SOURCE_PREFIX);
        List<List<String>> created = Collections.singletonList(result.get(1));
        String output1 = FeatureLearner.formatForSaving(FeatureLearner.processedInstances,
                created);
        //   individual.printIndividualForHumans(state, 0);

        try {
            FeatureLearner.writeOut(created, output1, "gp%s%dCreatedF-" + individual.fitness.fitness());
        } catch (IOException e) {
            e.printStackTrace();
        }
//

        double[][] outputs = problem.getOutputsForSaving(state, 0, individual);

        return formatAsCSV(processedInstances, outputs, FEAT_INDEX, SOURCE_PREFIX);
    }

    public static List<List<String>> formatAsCSV(List<Instance> processedInstances, double[][] outputs, int[] featIndex, String sourcePrefix) {
        StringBuilder sbAll = new StringBuilder();
        StringBuilder sbRF = new StringBuilder();
        LOG.printf("Created %d features\n", outputs.length);
        List<String> fileOutput = new ArrayList<>();
        List<String> rfFileOutput = new ArrayList<>();
        for (int fI : featIndex) {
            sbAll.append(String.format("F%s, ", fI));

        }
        sbAll.delete(sbAll.length() - 2, sbAll.length());

        for (int j = 0; j < outputs.length; j++) {
            sbAll.append(String.format(", F%s%c", sourcePrefix, FeatureCreatorStatistics.getCharToUse(j)));
            sbRF.append(String.format("F%s%c, ", sourcePrefix, FeatureCreatorStatistics.getCharToUse(j)));

        }
        sbRF.delete(sbRF.length() - 2, sbRF.length());

        fileOutput.add(sbAll.toString());
        rfFileOutput.add(sbRF.toString());
        for (int i = 0; i < processedInstances.size(); i++) {
            sbAll = new StringBuilder();
            sbRF = new StringBuilder();
            Instance instance = processedInstances.get(i);
            for (int fI : featIndex) {
                sbAll.append(instance.getFeatureValue(fI)).append(", ");

            }
            for (int j = 0; j < outputs.length; j++) {
                String format = String.format("%f, ", outputs[j][i]);
                sbAll.append(format);
                sbRF.append(format);
            }
            sbAll.delete(sbAll.length() - 2, sbAll.length());
            sbRF.delete(sbRF.length() - 2, sbRF.length());
            rfFileOutput.add(sbRF.toString());
            fileOutput.add(sbAll.toString());
        }
        //  fileOutput.forEach(LOG::println);
        List<List<String>> toReturn = new ArrayList<>();
        toReturn.add(fileOutput);
        toReturn.add(rfFileOutput);
        return toReturn;
    }

    static EvolutionState doGP() {
        String paramName = Main.CONFIG.getProperty("featureLearnParamFile");
        ParameterDatabase parameters;
        if (IN_A_JAR) {
            Util.LOG.println("Inside a jar, loading config internally.");
            InputStream resource = FeatureLearner.class.getClassLoader().getResourceAsStream(paramName.split("/gp/")[1]);
            Util.LOG.println(resource);
            try {
                parameters = Util.readECJParamsAsStream(resource);
            } catch (IOException e) {
                throw new Error(e);
            }
        } else {
            parameters = Evolve.loadParameterDatabase(new String[]{"-file", paramName});

        }

        FeatureLearnerProblem.normalisePostCreation = Main.CONFIG.getBoolean("normalisePostCreation");

        FeatureLearnerProblem.scalePostCreation = !FeatureLearnerProblem.normalisePostCreation && Main.CONFIG.getBoolean("scalePostCreation");
        if (FeatureLearnerProblem.normalisePostCreation) {
            Util.LOG.println("Normalising post-creation.");
        } else if (FeatureLearnerProblem.scalePostCreation) {
            Util.LOG.println("Scaling post-creation.");
        } else {
            Util.LOG.println("Not scaling OR normalising post-creation");
        }
        FeatureLearnerProblem.roundPostCreation = CONFIG.getBoolean("roundPostCreation");
        if (FeatureLearnerProblem.roundPostCreation) {
            Util.LOG.println("Rounding to 5dp post-creation.");
        }
        int numtrees;
        String num_trees = Main.CONFIG.getProperty("numtrees");
        switch (num_trees) {
            case "sqrt":
                numtrees = (int) Math.ceil(Math.sqrt(MULTIVARIATE ? multipleXVals.length : xVals.length));
                Util.LOG.println("Using sqrt(m) as number of trees");
                break;
            case "cubert":
                numtrees = (int) Math.ceil(Math.pow(MULTIVARIATE ? multipleXVals.length : xVals.length, 1d / 3d));
                Util.LOG.println("Using cubert(m) as number of trees");
                break;
            case "log2class":
                numtrees = (int) Math.ceil(Math.log(CONFIG.getInt("numclasses") / Math.log(2)));
                Util.LOG.println("Using log2(n)+1 as number of trees");
                break;
            case "log2class+1":
                numtrees = (int) Math.ceil(Math.log(CONFIG.getInt("numclasses") / Math.log(2))) + 1;
                Util.LOG.println("Using log2(n)+1 as number of trees");
                break;
            default:
                numtrees = Integer.parseInt(num_trees);

        }
        if (numtrees != -1) {
            Util.LOG.printf("%d trees\n", numtrees);
            parameters.set(new Parameter("pop.subpop.0.species.ind.numtrees"), Integer.toString(numtrees));
            for (int i = 0; i < numtrees; i++) {
                parameters.set(new Parameter("pop.subpop.0.species.ind.tree." + i), "ec.gp.GPTree");
                parameters.set(new Parameter("pop.subpop.0.species.ind.tree." + i + ".tc"), "tc0");
            }
//            if (paramName.contains("LNC")) {
//                try {
//                    System.out.println("LNC MI");
//                    baseMI = CallRcode.computeMI(multipleXVals, multipleXVals);
//                } catch (IOException | InterruptedException e) {
//                    e.printStackTrace();
//                }
//            } else {
//                baseMI = MutualInformationMap.getMultiVarMutualInformationVers2(multipleXVals, multipleXVals);
//            }
//            baseMultiInfo = MutualInformationMap.getMultiInformation(multipleXVals, false);
        }
        LOG.println("Base MI: " + baseMI);
        LOG.println("Base MultiInfo: " + baseMultiInfo);

        if (Main.CONFIG.containsKey("treeDepth")) {
            int treeDepth = Main.CONFIG.getInt("treeDepth");
            Util.LOG.printf("Tree depth: %d\n", treeDepth);

            parameters.set(new Parameter("gp.koza.xover.maxdepth"), "" + treeDepth);
            parameters.set(new Parameter("gp.koza.xover.maxsize"), "" + treeDepth);

            parameters.set(new Parameter("gp.koza.grow.max-depth"), "" + treeDepth);
            parameters.set(new Parameter("gp.koza.full.max-depth"), "" + treeDepth);
            parameters.set(new Parameter("gp.koza.half.max-depth"), "" + treeDepth);

        }


        int threads;
        int processors = Runtime.getRuntime().availableProcessors();

        if (CONFIG.containsKey("numthreads")) {
            String numthreads = CONFIG.getProperty("numthreads");
            if ("half".equals(numthreads)) {
                threads = processors / 2;
            } else {
                threads = CONFIG.getInt("numthreads");
            }
            Util.LOG.printf("Using %d threads as per config\n", threads);

        } else {
            threads = Math.max(processors - 1, 1);
        }
        Util.LOG.printf("%d processors, using %d threads\n", processors, threads);

        parameters.set(new Parameter("evalthreads"), "" + threads);
        // parameters.set(new Parameter("breedthreads"), "" + threads);

        parameters.set(new Parameter("stat.file"), Util.LOG.ECJ_OUT + Main.RUN + "F" + SOURCE_PREFIX);
        parameters.set(new Parameter("stat.front"), Util.LOG.PARETO_OUT);

        int seed = ThreadLocalRandom.current().nextInt();
        for (int i = 0; i < threads; i++) {
            parameters.set(new Parameter("seed." + i), Integer.toString(seed));
            seed++;
        }


        EvolutionState state = Evolve.initialize(parameters, 0);
        state.run(EvolutionState.C_STARTED_FRESH);
        state.output.close();
        return state;

    }

    public static void instancesToCSV(List<Instance> instances, int numOriginalFeatures) {
        List<String> fileOutput = new ArrayList<>();
        StringBuilder allF = new StringBuilder();

        for (int i = 0; i < numOriginalFeatures; i++) {
            allF.append(String.format("F%s, ", i));
        }
        allF.delete(allF.length() - 2, allF.length());
        fileOutput.add(allF.toString());

        for (int i = 0; i < instances.size(); i++) {
            allF = new StringBuilder();
            for (int j = 0; j < numOriginalFeatures; j++) {
                allF.append(instances.get(i).getFeatureValue(j)).append(", ");
            }
            allF.delete(allF.length() - 2, allF.length());

            fileOutput.add(allF.toString());


        }
        List<List<String>> allLines = Collections.singletonList(fileOutput);
        String allFS = formatForSaving(instances, allLines);
        try {
            writeOut(allLines, allFS, "%s");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void writeWekaToFile(List<Instance> instances, Instances pcaed, int numOriginalFeatures) throws IOException {

        List<String> fileOutput = new ArrayList<>();
        List<String> pcaFileOutput = new ArrayList<>();
        StringBuilder allF = new StringBuilder();
        StringBuilder pcaF = new StringBuilder();
        for (int i = 0; i < numOriginalFeatures; i++) {
            allF.append(String.format("F%s, ", i));
        }
        for (int i = 0; i < pcaed.numAttributes() - 1; i++) {
            allF.append(pcaed.attribute(i).name()).append(", ");
            pcaF.append(pcaed.attribute(i).name()).append(", ");
        }
        allF.delete(allF.length() - 2, allF.length());
        pcaF.delete(pcaF.length() - 2, pcaF.length());
        //allF.append("class\n");
        // pcaF.append("class\n");

        fileOutput.add(allF.toString());
        pcaFileOutput.add(pcaF.toString());

        //Can we assume weka preserves order...? Sure.
        for (int i = 0; i < instances.size(); i++) {
            allF = new StringBuilder();
            pcaF = new StringBuilder();
            for (int j = 0; j < numOriginalFeatures; j++) {
                allF.append(instances.get(i).getFeatureValue(j)).append(", ");
            }
            weka.core.Instance pcaInstance = pcaed.get(i);
            for (int j = 0; j < pcaInstance.numAttributes() - 1; j++) {
                String pcaStr = String.format("%f, ", pcaInstance.value(j));
                allF.append(pcaStr);
                pcaF.append(pcaStr);
            }
            allF.delete(allF.length() - 2, allF.length());
            pcaF.delete(pcaF.length() - 2, pcaF.length());

            fileOutput.add(allF.toString());
            pcaFileOutput.add(pcaF.toString());


        }
        fileOutput.forEach(LOG::println);
        List<List<String>> allLines = Collections.singletonList(fileOutput);
        String allFS = formatForSaving(instances, allLines);
        writeOut(allLines, allFS, "pca-added-%s%d");
        allLines = Collections.singletonList(pcaFileOutput);
        allFS = formatForSaving(instances, allLines);
        writeOut(allLines, allFS, "pca-only-%s%d");

    }

    public static class FLResult {

        final List<List<String>> csv;
        final EvolutionState state;

        public FLResult(List<List<String>> csv, EvolutionState state) {

            this.csv = csv;
            this.state = state;
        }
    }
}
