package eval;

import classifier.Classifier;
import model.Email;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Locale;

/**
 * Evaluator: tests a Classifier on unseen emails, builds a 2x2 confusion matrix,
 * computes accuracy, precision, recall, F1-score, and measures runtime.
 *
 * Confusion Matrix layout (int[2][2]):
 *   matrix[0][0] = True Positives  (Actual: SPAM, Predicted: SPAM)
 *   matrix[0][1] = False Positives (Actual: HAM,  Predicted: SPAM)
 *   matrix[1][0] = False Negatives (Actual: SPAM, Predicted: HAM)
 *   matrix[1][1] = True Negatives  (Actual: HAM,  Predicted: HAM)
 */
public class Evaluator {

    public static class EvaluationResult {
        public final String classifierName;
        public final int[][] confusionMatrix;
        public final int truePositives;
        public final int falsePositives;
        public final int falseNegatives;
        public final int trueNegatives;
        public final double accuracy;
        public final double precision;
        public final double recall;
        public final double f1Score;
        public final long trainTimeMs;
        public final long predictTimeMs;

        public EvaluationResult(
                String classifierName,
                int[][] confusionMatrix,
                double accuracy,
                double precision,
                double recall,
                double f1Score,
                long trainTimeMs,
                long predictTimeMs
        ) {
            this.classifierName = classifierName;
            this.confusionMatrix = confusionMatrix;
            this.truePositives = confusionMatrix[0][0];
            this.falsePositives = confusionMatrix[0][1];
            this.falseNegatives = confusionMatrix[1][0];
            this.trueNegatives = confusionMatrix[1][1];
            this.accuracy = accuracy;
            this.precision = precision;
            this.recall = recall;
            this.f1Score = f1Score;
            this.trainTimeMs = trainTimeMs;
            this.predictTimeMs = predictTimeMs;
        }
    }

    /**
     * Trains and evaluates a classifier, capturing both training and prediction times.
     */
    public EvaluationResult trainAndEvaluate(Classifier classifier, List<Email> trainData, List<Email> testData) {
        long trainStart = System.nanoTime();
        classifier.train(trainData);
        long trainTimeMs = (System.nanoTime() - trainStart) / 1_000_000;

        return evaluate(classifier, testData, trainTimeMs);
    }

    /**
     * Evaluates an already-trained classifier on testData.
     */
    public EvaluationResult evaluate(Classifier classifier, List<Email> testData) {
        return evaluate(classifier, testData, -1L);
    }

    private EvaluationResult evaluate(Classifier classifier, List<Email> testData, long trainTimeMs) {
        int[][] matrix = new int[2][2];

        long predStart = System.nanoTime();
        for (Email email : testData) {
            boolean actualSpam = email.isSpam();
            boolean predSpam = classifier.predict(email);

            if (actualSpam && predSpam) {
                matrix[0][0]++; // TP
            } else if (!actualSpam && predSpam) {
                matrix[0][1]++; // FP
            } else if (actualSpam && !predSpam) {
                matrix[1][0]++; // FN
            } else {
                matrix[1][1]++; // TN
            }
        }
        long predictTimeMs = (System.nanoTime() - predStart) / 1_000_000;

        int tp = matrix[0][0];
        int fp = matrix[0][1];
        int fn = matrix[1][0];
        int tn = matrix[1][1];
        int total = tp + fp + fn + tn;

        double accuracy = total > 0 ? (double) (tp + tn) / total : 0.0;
        double precision = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
        double recall = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
        double f1Score = (precision + recall) > 0
                ? 2.0 * (precision * recall) / (precision + recall)
                : 0.0;

        return new EvaluationResult(
                classifier.getName(),
                matrix,
                accuracy,
                precision,
                recall,
                f1Score,
                trainTimeMs,
                predictTimeMs
        );
    }

    /**
     * Prints a formatted evaluation report with the 2x2 confusion matrix,
     * accuracy, precision, recall, F1-score, and timing.
     */
    public void printReport(EvaluationResult r) {
        System.out.println("==============================================");
        System.out.println("Evaluation Report: " + r.classifierName);
        System.out.println("==============================================");
        System.out.println("Confusion Matrix (Rows = Actual, Cols = Predicted):");
        System.out.println("                   Pred: SPAM     Pred: HAM");
        System.out.printf(Locale.US, "Actual: SPAM       %6d (TP)    %6d (FN)%n",
                r.truePositives, r.falseNegatives);
        System.out.printf(Locale.US, "Actual: HAM        %6d (FP)    %6d (TN)%n",
                r.falsePositives, r.trueNegatives);
        System.out.println("----------------------------------------------");
        System.out.printf(Locale.US, "Accuracy:          %.2f%% (%.4f)%n", r.accuracy * 100.0, r.accuracy);
        System.out.printf(Locale.US, "Precision:         %.2f%% (%.4f)%n", r.precision * 100.0, r.precision);
        System.out.printf(Locale.US, "Recall:            %.2f%% (%.4f)%n", r.recall * 100.0, r.recall);
        System.out.printf(Locale.US, "F1-Score:          %.2f%% (%.4f)%n", r.f1Score * 100.0, r.f1Score);
        if (r.trainTimeMs >= 0) {
            System.out.printf(Locale.US, "Train Time:        %d ms%n", r.trainTimeMs);
        }
        System.out.printf(Locale.US, "Predict Time:      %d ms%n", r.predictTimeMs);
        System.out.println("==============================================\n");
    }

    /**
     * Exports all classifier results to results.csv for generating charts in Excel/Sheets.
     */
    public void exportToCsv(List<EvaluationResult> results, String filePath) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            writer.println("Classifier,TP,FP,FN,TN,Accuracy,Precision,Recall,F1,TrainTimeMs,PredictTimeMs");
            for (EvaluationResult r : results) {
                writer.printf(Locale.US, "%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%d%n",
                        escapeCsv(r.classifierName),
                        r.truePositives,
                        r.falsePositives,
                        r.falseNegatives,
                        r.trueNegatives,
                        r.accuracy,
                        r.precision,
                        r.recall,
                        r.f1Score,
                        r.trainTimeMs,
                        r.predictTimeMs
                );
            }
            System.out.println("Saved evaluation results to: " + filePath);
        } catch (IOException e) {
            System.err.println("Error writing CSV: " + e.getMessage());
        }
    }

    private String escapeCsv(String value) {
        if (value == null) return "";
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }
}