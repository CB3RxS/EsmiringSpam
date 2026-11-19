package data;

import model.Email;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * DataSplitter: shuffles the emails and cuts them into a TRAINING part
 * (the algorithm learns from it) and a TESTING part (the algorithm is
 * graded on it, and has never seen these emails before).
 *
 * The seed makes the shuffle repeatable: same seed = same split every run.
 * Every algorithm must use the SAME split so the comparison is fair.
 */
public class DataSplitter {

    private final List<Email> train;
    private final List<Email> test;

    public DataSplitter(List<Email> emails, double trainRatio, long seed) {
        List<Email> shuffled = new ArrayList<>(emails); // copy, keep original order intact
        Collections.shuffle(shuffled, new Random(seed));

        int cut = (int) (shuffled.size() * trainRatio);
        this.train = new ArrayList<>(shuffled.subList(0, cut));
        this.test = new ArrayList<>(shuffled.subList(cut, shuffled.size()));
    }

    public List<Email> getTrain() { return train; }
    public List<Email> getTest()  { return test; }
}