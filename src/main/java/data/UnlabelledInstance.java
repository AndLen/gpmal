package data;

import java.util.List;

/**
 * Created by lensenandr on 7/04/16.
 */
public class UnlabelledInstance extends Instance {
    public UnlabelledInstance(double[] featureValues) {
        super(featureValues, null, -1);
    }


    @Override
    public boolean equals(Object o) {
        throw new UnsupportedOperationException();

    }

    @Override
    public int hashCode() {
        throw new UnsupportedOperationException();
    }

}
