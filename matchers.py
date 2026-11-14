"""Matchers: each one gives a similarity score in [0, 1] to a pair of product names."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfMatcher:
    """Baseline: cosine similarity between the TF-IDF vectors of the two names."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), lowercase=True, norm="l2")

    def fit(self, names):
        """Learn how rare each word is, from all product names in both catalogs."""
        self.vectorizer.fit(names)
        return self

    def score(self, left_names, right_names):
        """Return one similarity score per (left, right) pair."""
        left = self.vectorizer.transform(left_names)
        right = self.vectorizer.transform(right_names)
        # Vectors are already normalised to length 1, so the dot product is the cosine.
        return np.asarray(left.multiply(right).sum(axis=1)).ravel()



class EmbeddingMatcher:
    """Cosine similarity between sentence embeddings of the two names (Hugging Face model)."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, names):
        """Nothing to learn here: the model is already trained on millions of sentences."""
        return self

    def score(self, left_names, right_names):
        """Return one similarity score per (left, right) pair."""
        left = self.model.encode(list(left_names), normalize_embeddings=True, batch_size=64)
        right = self.model.encode(list(right_names), normalize_embeddings=True, batch_size=64)
        # Both are normalised to length 1, so the dot product is the cosine (same as TF-IDF).
        return (left * right).sum(axis=1)


class HybridMatcher:
    """Weighted average of two matchers: TF-IDF sees characters, the fine-tuned model sees meaning."""

    def __init__(self, first, second, weight=0.5):
        self.first = first
        self.second = second
        self.weight = weight   # how much the first matcher counts (0.5 = equal)

    def fit(self, names):
        self.first.fit(names)
        self.second.fit(names)
        return self

    def score(self, left_names, right_names):
        a = self.first.score(left_names, right_names)
        b = self.second.score(left_names, right_names)
        return self.weight * a + (1 - self.weight) * b

if __name__ == "__main__":
    from data import load_catalogs, load_pairs

    table_a, table_b = load_catalogs()
    valid = load_pairs("valid", table_a, table_b)

    matcher = TfidfMatcher().fit(list(table_a["name"]) + list(table_b["name"]))
    valid["score"] = matcher.score(valid["left_name"], valid["right_name"])

    print(valid.groupby("label")["score"].describe().round(3))
    examples = valid.sort_values("score", ascending=False)
    print(examples[["left_name", "right_name", "label", "score"]].head(5).to_string())
