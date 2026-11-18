import data.DataLoader;
import data.Preprocessor;
import model.Email;

import java.util.List;

/**
 * Main: a quick test of the Week 1 foundation.
 * Run it with the path to your dataset, e.g.:  data/emails.csv
 */
public class Main {

    public static void main(String[] args) throws Exception {
        String path = args.length > 0 ? args[0] : "data/emails.csv";

        // 1. Load
        DataLoader loader = new DataLoader();
        List<Email> emails = loader.loadEmails(path);

        // 2. Clean
        Preprocessor preprocessor = new Preprocessor();
        preprocessor.cleanAll(emails);

        // 3. Dataset summary (use these numbers in your report!)
        int spamCount = 0;
        for (Email e : emails) {
            if (e.isSpam()) spamCount++;
        }
        int hamCount = emails.size() - spamCount;
        System.out.println("Total emails: " + emails.size());
        System.out.println("Spam: " + spamCount + "  Ham: " + hamCount);
        System.out.printf("Spam ratio: %.1f%%%n", 100.0 * spamCount / emails.size());

        // 4. Before/after examples (use these in the Preprocessing section!)
        System.out.println("\n--- Before and after cleaning (first 3 emails) ---");
        for (int i = 0; i < Math.min(3, emails.size()); i++) {
            Email e = emails.get(i);
            System.out.println("BEFORE: " + e);
            System.out.println("AFTER : " + e.getTokens());
            System.out.println();
        }
    }
}
