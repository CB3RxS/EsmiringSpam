import classifier.Classifier;
import classifier.NaiveBayes;
import data.DataLoader;
import data.DataSplitter;
import data.Preprocessor;
import model.Email;

import java.util.List;

/**
 * Main: loads the data, splits it, trains Naive Bayes, and prints a quick accuracy.
 * (The Evaluator, coming next, will replace the quick accuracy check below.)
 */
public class Main {

    public static void main(String[] args) throws Exception {
        String path = args.length > 0 ? args[0] : "data/emails.csv";

        // 1. Load  (if your columns are text,spam use: loader.loadEmails(path, 1, 0))
        DataLoader loader = new DataLoader();
        List<Email> emails = loader.loadEmails(path, 1, 0);

        // 2. Clean
        new Preprocessor().cleanAll(emails);

        int spamCount = 0;
        for (Email e : emails) if (e.isSpam()) spamCount++;
        System.out.println("Total emails: " + emails.size());
        System.out.println("Spam: " + spamCount + "  Ham: " + (emails.size() - spamCount));
        System.out.printf("Spam ratio: %.1f%%%n", 100.0 * spamCount / emails.size());

        // 3. Split 80% train / 20% test, fixed seed 42
        DataSplitter split = new DataSplitter(emails, 0.8, 42);
        System.out.println("Train: " + split.getTrain().size()
                + "  Test: " + split.getTest().size());

        // 4. Train and test
        Classifier model = new NaiveBayes();
        model.train(split.getTrain());

        int correct = 0;
        for (Email e : split.getTest()) {
            if (model.predict(e) == e.isSpam()) correct++;
        }
        System.out.printf("%s accuracy: %.2f%%%n", model.getName(),
                100.0 * correct / split.getTest().size());
    }
}