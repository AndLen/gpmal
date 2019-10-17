package other;

import com.github.benmanes.caffeine.cache.Cache;
import featureLearn.ComputeFitness;

import java.util.concurrent.atomic.AtomicInteger;

import static featureLearn.FeatureLearnerProblem.createCache;

public class FitnessCache {
    private final Cache<Long, Double> CACHED_FITNESS = createCache(100_000_000);
    private AtomicInteger accesses = new AtomicInteger(0);


    public Double getFitness(long hashcode, ComputeFitness computeFitness, double[][] outputs) {
        Double fitness = CACHED_FITNESS.get(hashcode, a -> computeFitness.measureFitness(outputs));
        int curr = accesses.incrementAndGet();
        if (curr % 1000 == 0) {
            System.out.printf("Cache Size: %d\n", CACHED_FITNESS.asMap().keySet().size());
        }
        return fitness;
    }
}
