/*
  Copyright 2006 by Sean Luke
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/


package gp;

import ec.EvolutionState;
import ec.Problem;
import ec.gp.ADFStack;
import ec.gp.GPData;
import ec.gp.GPIndividual;
import ec.gp.GPNode;
import other.Util;

public class Sigmoid extends GPNode {
    public String toString() {
        return "sig";
    }

    public int expectedChildren() {
        return 1;
    }

    public void eval(final EvolutionState state,
                     final int thread,
                     final GPData input,
                     final ADFStack stack,
                     final GPIndividual individual,
                     final Problem problem) {
        DoubleData rd = ((DoubleData) (input));

        children[0].eval(state, thread, input, stack, individual, problem);
        rd.val = Util.sigmoid(rd.val);
    }
}

