package featureLearn;

@FunctionalInterface
public interface ComputeFitness {
    double measureFitness(double[][] outputs);
}
