package data;

import model.Email;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Preprocessor: turns messy email text into a clean list of words (tokens).
 *
 * Example:
 *   Input : "FREE!!! Click http://win.com NOW to claim your prize"
 *   Output: [free, click, urltoken, now, claim, prize]
 *
 * Steps (explain each one in your report):
 *   1. Lowercase everything         -> "FREE" and "free" become the same word
 *   2. Replace links with "urltoken" -> links are a strong spam/phishing clue
 *   3. Remove punctuation/symbols    -> "free!!!" becomes "free"
 *   4. Split into words (tokenize)
 *   5. Drop stop words and tiny words -> "the", "is", "a" don't help detection
 */
public class Preprocessor {

    // HashSet = O(1) lookup, so "is this a stop word?" is very fast.
    private final Set<String> stopWords = new HashSet<>(Arrays.asList(
            "the", "is", "am", "are", "was", "were", "be", "been", "a", "an",
            "and", "or", "but", "if", "of", "to", "in", "on", "at", "for",
            "with", "as", "by", "it", "its", "this", "that", "these", "those",
            "i", "you", "he", "she", "we", "they", "me", "my", "your", "our",
            "from", "so", "do", "does", "did", "have", "has", "had", "will",
            "would", "can", "could", "not", "no", "just", "than", "then"
    ));

    /** Cleans one piece of text and returns its tokens. */
    public List<String> clean(String text) {
        // Step 1: lowercase
        String result = text.toLowerCase();

        // Step 2: replace web links with one special word
        result = result.replaceAll("(https?://\\S+|www\\.\\S+)", " urltoken ");

        // Step 3: replace anything that is not a letter or digit with a space
        result = result.replaceAll("[^a-z0-9]+", " ");

        // Step 4: split on spaces
        String[] words = result.trim().split("\\s+");

        // Step 5: keep only useful words
        List<String> tokens = new ArrayList<>();
        for (String word : words) {
            if (word.length() < 2) continue;          // skip empty/1-letter words
            if (stopWords.contains(word)) continue;   // skip stop words
            tokens.add(word);
        }
        return tokens;
    }

    /** Cleans every email in the list and stores the tokens inside each Email. */
    public void cleanAll(List<Email> emails) {
        for (Email email : emails) {
            email.setTokens(clean(email.getRawText()));
        }
    }
}
