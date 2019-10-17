package tests.featureLearn;

import featureLearn.FeatureLearner;
import tests.Tests;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by lensenandr on 28/11/16.
 */
public class FLNeighboursFG extends Tests {

    public List<String> getTestConfig() {
        Tests.main = FeatureLearner.class;

        return new ArrayList<>(Arrays.asList(
                "preprocessing=scale", "logPrefix=fLNeighboursFG/", "treeDepth=8", "featureMin=0", "featureMax=1", "numtrees=2", "normalisePostCreation=false", "scalePostCreation=false", "roundPostCreation=true", "featureLearnParamFile=src/main/java/gp/flNeighboursFG.params", "doNNs=false"
                // "fitnessFunction=clusterFitness.LocumFitness",
        ));
    }
}
