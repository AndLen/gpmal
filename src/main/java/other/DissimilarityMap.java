package other;

import data.Instance;

/**
 * Created by lensenandr on 14/06/16.
 */
public interface DissimilarityMap {
    double getDissim(Instance i1, Instance i2);

    double averageDissim();
}
