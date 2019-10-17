package tests.featureLearn;

import featureLearn.FeatureLearner;
import tests.Tests;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by lensenandr on 28/11/16.
 */
public class FLNeighbours extends Tests {

    public List<String> getTestConfig() {
        Tests.main = FeatureLearner.class;

        return new ArrayList<>(Arrays.asList(
                "preprocessing=scale", "logPrefix=fLNeighbours/", "treeDepth=8", "featureMin=0", "featureMax=1", "numtrees=2", "normalisePostCreation=false", "scalePostCreation=false", "roundPostCreation=true","featureLearnParamFile=src/main/java/gp/flNeighbours.params"
                // "fitnessFunction=clusterFitness.LocumFitness",
        ));
    }
}
