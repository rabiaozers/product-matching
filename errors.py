"""Error analysis: where does the embedding model go wrong, and does TF-IDF get those right?"""
import pandas as pd

from data import load_catalogs, load_pairs
from evaluate import best_threshold
from matchers import EmbeddingMatcher, TfidfMatcher

table_a, table_b = load_catalogs()
valid = load_pairs("valid", table_a, table_b)
test = load_pairs("test", table_a, table_b)
all_names = list(table_a["name"]) + list(table_b["name"])

# Score valid and test with both matchers; each gets its own threshold from valid.
for short, matcher in [("tfidf", TfidfMatcher()), ("emb", EmbeddingMatcher())]:
    matcher.fit(all_names)
    t = best_threshold(valid["label"], matcher.score(valid["left_name"], valid["right_name"]))
    test[short + "_score"] = matcher.score(test["left_name"], test["right_name"])
    test[short + "_pred"] = (test[short + "_score"] > t).astype(int)

pd.set_option("display.max_colwidth", 55)
pd.set_option("display.width", 200)
cols = ["left_name", "right_name", "emb_score", "tfidf_score"]

emb_fp = test[(test["emb_pred"] == 1) & (test["label"] == 0)][cols + ["label"]]
emb_fn = test[(test["emb_pred"] == 0) & (test["label"] == 1)][cols + ["label"]]

print("\n=== Embedding false alarms (FP), highest scores first ===")
print(emb_fp.sort_values("emb_score", ascending=False)[cols].head(8).round(2).to_string())

print("\n=== Embedding misses (FN), lowest scores first ===")
print(emb_fn.sort_values("emb_score")[cols].head(8).round(2).to_string())

print("\n=== Who is right? ===")
print(pd.crosstab(test["tfidf_pred"] == test["label"], test["emb_pred"] == test["label"],
                  rownames=["TF-IDF right"], colnames=["Embedding right"]))
