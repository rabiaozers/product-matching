"""Compare the original fine-tuned model with the one trained with extra easy negatives.

For each model: /match quality (test F1, alone and in the hybrid) and
/search quality (recall@k, alone and in the hybrid). Weight and threshold
are always chosen on valid; test is measured once. Writes nothing.
"""
import numpy as np

from data import load_catalogs, load_pairs
from evaluate import best_threshold, precision_recall_f1
from matchers import EmbeddingMatcher, TfidfMatcher
from search import CatalogSearch, embedding_vectors, hybrid_vectors, recall_at_k, tfidf_vectors

MODELS = {"original": "models/minilm-abtbuy", "+ easy negatives": "models/minilm-abtbuy-neg"}

table_a, table_b = load_catalogs()
valid = load_pairs("valid", table_a, table_b)
test = load_pairs("test", table_a, table_b)
matches = test[test["label"] == 1]
tfidf = TfidfMatcher().fit(list(table_a["name"]) + list(table_b["name"]))
a_valid = tfidf.score(valid["left_name"], valid["right_name"])
a_test = tfidf.score(test["left_name"], test["right_name"])


def test_f1(valid_scores, test_scores):
    """Threshold from valid, F1 on test."""
    t = best_threshold(valid["label"], valid_scores)
    return precision_recall_f1(test["label"], (test_scores > t).astype(int))[2]


def best_weight(b_valid):
    """Hybrid weight with the highest valid F1."""
    best_f1, best_w = -1.0, None
    for w in np.arange(0.0, 1.01, 0.1):
        scores = w * a_valid + (1 - w) * b_valid
        t = best_threshold(valid["label"], scores)
        f1 = precision_recall_f1(valid["label"], (scores > t).astype(int))[2]
        if f1 > best_f1:
            best_f1, best_w = f1, w
    return best_w


rows = []
for name, model_dir in MODELS.items():
    fine = EmbeddingMatcher(model_dir)
    b_valid = fine.score(valid["left_name"], valid["right_name"])
    b_test = fine.score(test["left_name"], test["right_name"])
    w = best_weight(b_valid)

    match_alone = test_f1(b_valid, b_test)
    match_hybrid = test_f1(w * a_valid + (1 - w) * b_valid, w * a_test + (1 - w) * b_test)

    emb = embedding_vectors(fine.model)
    search_alone = recall_at_k(CatalogSearch(emb, table_b["name"]), matches, table_b["id"].values)
    hyb = hybrid_vectors(tfidf_vectors(tfidf.vectorizer), emb, w)
    search_hybrid = recall_at_k(CatalogSearch(hyb, table_b["name"]), matches, table_b["id"].values)
    rows.append((name, w, match_alone, match_hybrid, search_alone, search_hybrid))
    print("done:", name)

print(f"\n{'model':18s} {'w':>4s} | {'match F1':>9s} {'hybrid F1':>9s} | "
      f"{'search R@1':>10s} {'R@5':>6s} | {'hybrid R@1':>10s} {'R@5':>6s}")
for name, w, ma, mh, sa, sh in rows:
    print(f"{name:18s} {w:4.1f} | {ma:9.3f} {mh:9.3f} | {sa[1]:10.3f} {sa[5]:6.3f} | {sh[1]:10.3f} {sh[5]:6.3f}")
