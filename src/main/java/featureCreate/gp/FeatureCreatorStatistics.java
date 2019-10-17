package featureCreate.gp;

import ec.Statistics;

/**
 * Created by lensenandr on 8/07/16.
 */
public class FeatureCreatorStatistics extends Statistics {


    public static char getCharToUse(int i) {
        return i < 26 ? (char) ('a' + i) : (char) ('α' + (i - 26));
    }

}
