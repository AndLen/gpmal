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

import java.util.Arrays;

public abstract class KAdd extends GPNode {

    public String toString() {
        return getK() + "+";
    }

    @Override
    public int expectedChildren() {
        return getK();
    }

    protected abstract int getK();


    @Override
    public void eval(EvolutionState state, int thread, GPData input, ADFStack stack, GPIndividual individual, Problem problem) {
        DoubleData doubleData = new DoubleData();
        double[] childRes = new double[children.length];
        for (int i = 0; i < children.length; i++) {
            GPNode child = children[i];
            child.eval(state, thread, doubleData, stack, individual, problem);
            childRes[i] = doubleData.val;
        }
        ((DoubleData) input).val = performOperation(childRes);
    }

    protected double performOperation(double... childen) {
        return Arrays.stream(childen).sum();
    }
}

