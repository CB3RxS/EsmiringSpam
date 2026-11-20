package classifier;

import model.Email;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Converts cleaned email tokens into normalized TF-IDF feature vectors.
 * The vocabulary and IDF values are learned from TRAINING DATA ONLY.
 */
final class TextVectorizer {

    private static final int MAX_FEATURES = 3000;

    private final Map<String, Integer> wordToIndex = new LinkedHashMap<>();
    private double[] inverseDocumentFrequency;

    void fit(List<Email> trainingData) {
        Map<String, Integer> documentFrequency = new HashMap<>();

        for (Email email : trainingData) {
            Map<String, Boolean> seen = new HashMap<>();
            for (String token : email.getTokens()) {
                if (seen.put(token, true) == null) {
                    documentFrequency.merge(token, 1, Integer::sum);
                }
            }
        }

        List<Map.Entry<String, Integer>> rankedWords =
                new ArrayList<>(documentFrequency.entrySet());

        rankedWords.sort(
                Comparator.<Map.Entry<String, Integer>>comparingInt(Map.Entry::getValue)
                        .reversed()
                        .thenComparing(Map.Entry::getKey)
        );

        wordToIndex.clear();
        int featureCount = Math.min(MAX_FEATURES, rankedWords.size());
        inverseDocumentFrequency = new double[featureCount];

        for (int i = 0; i < featureCount; i++) {
            Map.Entry<String, Integer> entry = rankedWords.get(i);
            wordToIndex.put(entry.getKey(), i);

            // IDF gives less weight to words appearing in almost every email.
            inverseDocumentFrequency[i] =
                    Math.log((1.0 + trainingData.size()) / (1.0 + entry.getValue())) + 1.0;
        }
    }

    Vector transform(Email email) {
        Map<Integer, Integer> termCounts = new HashMap<>();
        int totalKnownTerms = 0;

        for (String token : email.getTokens()) {
            Integer index = wordToIndex.get(token);
            if (index != null) {
                termCounts.merge(index, 1, Integer::sum);
                totalKnownTerms++;
            }
        }

        if (totalKnownTerms == 0) {
            return new Vector(new int[0], new double[0]);
        }

        int[] indices = new int[termCounts.size()];
        double[] values = new double[termCounts.size()];

        int position = 0;
        double squaredNorm = 0.0;

        // Sort feature indexes so sparse-vector dot products can use two pointers.
        List<Integer> sortedIndexes = new ArrayList<>(termCounts.keySet());
        sortedIndexes.sort(Integer::compareTo);

        for (int index : sortedIndexes) {
            double termFrequency =
                    (double) termCounts.get(index) / totalKnownTerms;
            double tfIdf = termFrequency * inverseDocumentFrequency[index];

            indices[position] = index;
            values[position] = tfIdf;
            squaredNorm += tfIdf * tfIdf;
            position++;
        }

        // L2 normalization makes cosine similarity efficient and comparable.
        double norm = Math.sqrt(squaredNorm);
        if (norm > 0.0) {
            for (int i = 0; i < values.length; i++) {
                values[i] /= norm;
            }
        }

        return new Vector(indices, values);
    }

    int getFeatureCount() {
        return inverseDocumentFrequency == null ? 0 : inverseDocumentFrequency.length;
    }

    /** Sparse vector: only non-zero feature indexes and their values are stored. */
    static final class Vector {
        final int[] indexes;
        final double[] values;

        Vector(int[] indexes, double[] values) {
            this.indexes = indexes;
            this.values = values;
        }

        /** Dot product of two sorted sparse vectors. */
        double dot(Vector other) {
            int i = 0;
            int j = 0;
            double result = 0.0;

            while (i < indexes.length && j < other.indexes.length) {
                if (indexes[i] == other.indexes[j]) {
                    result += values[i] * other.values[j];
                    i++;
                    j++;
                } else if (indexes[i] < other.indexes[j]) {
                    i++;
                } else {
                    j++;
                }
            }
            return result;
        }
    }
  
}
