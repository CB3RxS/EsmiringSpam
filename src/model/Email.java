package model;

import java.util.ArrayList;
import java.util.List;

/**
 * Email: a container for ONE email from the dataset.
 *
 * It stores three things:
 *   1. rawText  - the original email text, exactly as it was in the file
 *   2. spam     - the correct answer (true = spam, false = ham/not spam)
 *   3. tokens   - the cleaned words (filled in later by the Preprocessor)
 */
public class Email {

    private final String rawText;
    private final boolean spam;
    private List<String> tokens = new ArrayList<>();

    public Email(String rawText, boolean spam) {
        this.rawText = rawText;
        this.spam = spam;
    }

    public String getRawText() {
        return rawText;
    }

    public boolean isSpam() {
        return spam;
    }

    public List<String> getTokens() {
        return tokens;
    }

    public void setTokens(List<String> tokens) {
        this.tokens = tokens;
    }

    @Override
    public String toString() {
        String label = spam ? "SPAM" : "HAM ";
        String preview = rawText.length() > 60 ? rawText.substring(0, 60) + "..." : rawText;
        return "[" + label + "] " + preview.replace("\n", " ");
    }
}
