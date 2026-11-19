package classifier;

import model.Email;

import java.util.List;

/**
 * Classifier: the "contract" every algorithm must follow.
 * NaiveBayes, KNN and LogisticRegression will all implement it, so the
 * Evaluator and the GUI can use any of them without caring which one it is.
 */
public interface Classifier {

    /** Learn from emails whose correct answers (spam/ham) are known. */
    void train(List<Email> trainingData);

    /** Look at one email and answer: true = spam, false = ham. */
    boolean predict(Email email);

    /** Name shown in tables and in the GUI dropdown. */
    String getName();
}