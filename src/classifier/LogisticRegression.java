package classifier;

import model.Email;

import java.util.List;

/**
 * Binary Logistic Regression trained with batch gradient descent.
 *
 * Each TF-IDF feature has a weight. The weighted sum is passed through
 * the sigmoid function to estimate the probability that an email is spam.
 */
public class LogisticRegression implements Classifier {

    private static final int EPOCHS = 300;
    private static final double LEARNING_RATE = 5.0;
    private static final double L2_STRENGTH = 0.0001;
    private static final double CLASSIFICATION_THRESHOLD = 0.5;

    private final TextVectorizer vectorizer = new TextVectorizer();

    private double[] weights;
    private double bias;
    private boolean trained;

    @Override
    public void train(List<Email> trainingData) {
        if (trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty.");
        }

        vectorizer.fit(trainingData);
        weights = new double[vectorizer.getFeatureCount()];
        bias = 0.0;
        trained = false;

        int sampleCount = trainingData.size();

        // Convert the training emails once instead of rebuilding vectors every epoch.
        TextVectorizer.Vector[] trainingVectors =
                new TextVectorizer.Vector[sampleCount];
        double[] expectedLabels = new double[sampleCount];

        for (int i = 0; i < sampleCount; i++) {
            Email email = trainingData.get(i);
            trainingVectors[i] = vectorizer.transform(email);
            expectedLabels[i] = email.isSpam() ? 1.0 : 0.0;
        }

        // Batch gradient descent: each epoch updates weights using all examples.
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            double[] weightGradients = new double[weights.length];
            double biasGradient = 0.0;

            for (int sample = 0; sample < sampleCount; sample++) {
                TextVectorizer.Vector vector = trainingVectors[sample];
                double probability = sigmoid(predictScore(vector));
                double error = probability - expectedLabels[sample];

                biasGradient += error;

                // Sparse update: only features present in this email contribute.
                for (int i = 0; i < vector.indexes.length; i++) {
                    int featureIndex = vector.indexes[i];
                    weightGradients[featureIndex] += error * vector.values[i];
                }
            }

            double scale = 1.0 / sampleCount;
            for (int i = 0; i < weights.length; i++) {
                double gradient = weightGradients[i] * scale
                        + L2_STRENGTH * weights[i];
                weights[i] -= LEARNING_RATE * gradient;
            }

            bias -= LEARNING_RATE * biasGradient * scale;
        }

        trained = true;
    }

    @Override
    public boolean predict(Email email) {
        if (!trained) {
            throw new IllegalStateException(
                    "Train the Logistic Regression classifier before prediction."
            );
        }

        TextVectorizer.Vector vector = vectorizer.transform(email);
        double spamProbability = sigmoid(predictScore(vector));
        return spamProbability >= CLASSIFICATION_THRESHOLD;
    }

    /** Calculates bias + the sum of (feature value * corresponding weight). */
    private double predictScore(TextVectorizer.Vector vector) {
        double score = bias;

        for (int i = 0; i < vector.indexes.length; i++) {
            score += weights[vector.indexes[i]] * vector.values[i];
        }
        return score;
    }

    /** Converts any real-valued score into a probability between 0 and 1. */
    private double sigmoid(double value) {
        // Avoid numerical overflow for very large positive or negative scores.
        if (value >= 0.0) {
            double exp = Math.exp(-value);
            return 1.0 / (1.0 + exp);
        }

        double exp = Math.exp(value);
        return exp / (1.0 + exp);
    }

    @Override
    public String getName() {
        return "Logistic Regression";
    }
}
