package classifier;

import model.Email;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * NaiveBayes: decides spam or ham by comparing how likely the email's words
 * are in spam versus in ham.
 *
 * Training  = counting words in each class.        O(N * L)  (N emails, L words each)
 * Predicting = looking up each word of ONE email.  O(L)      (HashMap lookup is O(1))
 */
public class NaiveBayes implements Classifier {

    private final Map<String, Integer> spamWordCounts = new HashMap<>();
    private final Map<String, Integer> hamWordCounts = new HashMap<>();
    private final Set<String> vocabulary = new HashSet<>();

    private int spamTotalWords = 0;   // all word occurrences in spam emails
    private int hamTotalWords = 0;    // all word occurrences in ham emails
    private int spamEmails = 0;
    private int hamEmails = 0;

    @Override
    public void train(List<Email> trainingData) {
        spamWordCounts.clear();
        hamWordCounts.clear();
        vocabulary.clear();
        spamTotalWords = hamTotalWords = spamEmails = hamEmails = 0;

        for (Email email : trainingData) {
            Map<String, Integer> counts;
            if (email.isSpam()) { counts = spamWordCounts; spamEmails++; }
            else                { counts = hamWordCounts;  hamEmails++; }

            for (String word : email.getTokens()) {
                counts.merge(word, 1, Integer::sum); // add 1 to this word's count
                vocabulary.add(word);
                if (email.isSpam()) spamTotalWords++; else hamTotalWords++;
            }
        }
    }

    @Override
    public boolean predict(Email email) {
        int totalEmails = spamEmails + hamEmails;
        // Start with the "prior": how common is each class overall?
        // We use log() so we ADD small numbers instead of multiplying them
        // (multiplying hundreds of tiny numbers rounds to 0 on a computer).
        double spamScore = Math.log((double) spamEmails / totalEmails);
        double hamScore = Math.log((double) hamEmails / totalEmails);

        int vocabSize = vocabulary.size();
        for (String word : email.getTokens()) {
            if (!vocabulary.contains(word)) continue; // never seen in training: ignore

            // Laplace smoothing: add 1 so a word seen only in one class
            // does not make the other class's probability exactly 0.
            double pWordSpam = (spamWordCounts.getOrDefault(word, 0) + 1.0)
                    / (spamTotalWords + vocabSize);
            double pWordHam = (hamWordCounts.getOrDefault(word, 0) + 1.0)
                    / (hamTotalWords + vocabSize);

            spamScore += Math.log(pWordSpam);
            hamScore += Math.log(pWordHam);
        }
        return spamScore > hamScore;
    }

    @Override
    public String getName() {
        return "Naive Bayes";
    }
}