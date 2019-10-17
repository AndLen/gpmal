package gp;

import ec.EvolutionState;
import ec.Problem;
import ec.gp.*;

import java.util.Objects;

/**
 * Created by Andrew on 8/04/2015.
 */
public class NegativeDoubleConstantNode extends ERC {
    public double constant;

    @Override
    public String toString() {
        return "" + constant;
    }

    @Override
    public String encode() {
        return "" + constant;
    }

    @Override
    public int expectedChildren() {
        return 0;
    }

    @Override
    public void resetNode(EvolutionState state, int thread) {
        this.constant = (state.random[thread].nextDouble(true, true));
        if (state.random[thread].nextBoolean(0.5)) this.constant *= -1;

    }


    @Override
    public boolean nodeEquals(GPNode node) {
        return node instanceof NegativeDoubleConstantNode && ((NegativeDoubleConstantNode) node).constant == constant;
    }

    @Override
    public void eval(EvolutionState state, int thread, GPData input, ADFStack stack, GPIndividual individual, Problem problem) {
        ((DoubleData) input).val = constant;
    }

    public int nodeHashCode() {
        // a reasonable hash code
        return Objects.hash(this.getClass(), constant);
    }
}
