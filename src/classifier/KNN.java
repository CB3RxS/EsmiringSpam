package classifier;

import model.Email;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * k-Nearest Neighbors classifier using cosine distance over TF-IDF vectors.
 *
 * Training stores the labeled feature vectors.
 * Prediction finds the k closest training emails and uses majority voting.
 */
public class KNN implements Classifier {

    private final int k;
    private final TextVectorizer vectorizer = new TextVectorizer();
    private final List<TrainingExample> trainingExamples = new ArrayList<>();

    public KNN(int k) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be at least 1.");
        }
        this.k = k;
    }

    @Override
    public void train(List<Email> trainingData) {
        trainingExamples.clear();
        vectorizer.fit(trainingData);

        for (Email email : trainingData) {
            trainingExamples.add(new TrainingExample(
                    vectorizer.transform(email), email.isSpam()
            ));
        }
    }

    @Override
    public boolean predict(Email email) {
        if (trainingExamples.isEmpty()) {
            throw new IllegalStateException("Train the kNN classifier before prediction.");
        }

        TextVectorizer.Vector query = vectorizer.transform(email);
        if (query.indexes.length == 0) {
            return false; // No known words: use the conservative HAM prediction.
        }

        // The largest distance is kept at the head so it can be removed
        // whenever a closer neighbor is found.
        PriorityQueue<Neighbor> nearest = new PriorityQueue<>(
                Comparator.comparingDouble((Neighbor n) -> n.distance).reversed()
        );

        int neighborCount = Math.min(k, trainingExamples.size());

        for (TrainingExample example : trainingExamples) {
            double cosineSimilarity = query.dot(example.vector);
            double distance = 1.0 - cosineSimilarity;

            Neighbor candidate = new Neighbor(distance, example.spam);

            if (nearest.size() < neighborCount) {
                nearest.offer(candidate);
            } else if (distance < nearest.peek().distance) {
                nearest.poll();
                nearest.offer(candidate);
            }
        }

        int spamVotes = 0;
        int hamVotes = 0;
        double spamDistanceTotal = 0.0;
        double hamDistanceTotal = 0.0;

        while (!nearest.isEmpty()) {
            Neighbor neighbor = nearest.poll();
            if (neighbor.spam) {
                spamVotes++;
                spamDistanceTotal += neighbor.distance;
            } else {
                hamVotes++;
                hamDistanceTotal += neighbor.distance;
            }
        }

        if (spamVotes != hamVotes) {
            return spamVotes > hamVotes;
        }

        // Tie-break: choose the class whose selected neighbors are closer overall.
        return spamDistanceTotal < hamDistanceTotal;
    }

    @Override
    public String getName() {
        return "k-Nearest Neighbors (k=" + k + ")";
    }

    private static final class TrainingExample {
        final TextVectorizer.Vector vector;
        final boolean spam;

        TrainingExample(TextVectorizer.Vector vector, boolean spam) {
            this.vector = vector;
            this.spam = spam;
        }
    }

    private static final class Neighbor {
        final double distance;
        final boolean spam;

        Neighbor(double distance, boolean spam) {
            this.distance = distance;
            this.spam = spam;
        }
    }
}
