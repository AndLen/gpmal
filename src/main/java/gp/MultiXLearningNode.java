package gp;

import ec.EvolutionState;
import ec.Problem;
import ec.gp.*;
import featureLearn.FeatureLearner;
import featureLearn.FeatureLearnerProblem;

import java.util.Objects;

/**
 * Created by Andrew on 8/04/2015.
 */
public class MultiXLearningNode extends ERC {

    public int val;

    public int getVal() {
        return val;
    }

    @Override
    public String toString() {
        return "X" + val;
    }

    @Override
    public String encode() {
        return toString();
    }

    @Override
    public int expectedChildren() {
        return 0;
    }

    @Override
    public void resetNode(EvolutionState state, int thread) {
        val = state.random[thread].nextInt(FeatureLearner.NUM_SOURCE_MV);
    }


    @Override
    public boolean nodeEquals(GPNode node) {
        if (node instanceof MultiXLearningNode) {
            MultiXLearningNode multiXNode = (MultiXLearningNode) node;
            return multiXNode.val == val;
        } else return false;
    }

    @Override
    public void eval(EvolutionState state, int thread, GPData input, ADFStack stack, GPIndividual individual, Problem problem) {
        ((DoubleData) input).val = ((FeatureLearnerProblem) problem).xVals[this.val];
    }

    public int nodeHashCode() {
        // a reasonable hash code
        return Objects.hash(getClass(), val);
    }
    //    public GPNode clone(){

}
