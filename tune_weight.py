"""Choose the hybrid weight and threshold on the validation set, measure once on test, save the config.

This is the only script that writes models/hybrid_config.json; app.py and evaluate.py read it.
"""
import json

import numpy as np

from data import load_catalogs, load_pairs
from evaluate import best_threshold, precision_recall_f1
from matchers import EmbeddingMatcher, TfidfMatcher

MODEL_DIR = "models/minilm-abtbuy"
CONFIG_PATH = "models/hybrid_config.json"

table_a, table_b = load_catalogs()
valid = load_pairs("valid", table_a, table_b)
test = load_pairs("test", table_a, table_b)
all_names = list(table_a["name"]) + list(table_b["name"])

# The slow part: score every pair ONCE with each matcher.
tfidf = TfidfMatcher().fit(all_names)
fine = EmbeddingMatcher(MODEL_DIR)
a_valid = tfidf.score(valid["left_name"], valid["right_name"])
b_valid = fine.score(valid["left_name"], valid["right_name"])
a_test = tfidf.score(test["left_name"], test["right_name"])
b_test = fine.score(test["left_name"], test["right_name"])

# The fast part: try each weight, only the mixing changes.
best_f1, best_w, best_t = -1.0, None, None
for w in np.arange(0.0, 1.01, 0.1):
    scores = w * a_valid + (1 - w) * b_valid
    t = best_threshold(valid["label"], scores)
    _, _, f1 = precision_recall_f1(valid["label"], (scores > t).astype(int))
    print(f"weight={w:.1f}  threshold={t:.2f}  valid F1={f1:.3f}")

   
    if f1 > best_f1:
        best_f1 = f1
        best_w = w
        best_t = t


print(f"\nchosen on valid: weight={best_w:.1f}, threshold={best_t:.2f}, valid F1={best_f1:.3f}")

# Measure once on test with the chosen weight and threshold.
test_scores = best_w * a_test + (1 - best_w) * b_test
p, r, f1 = precision_recall_f1(test["label"], (test_scores > best_t).astype(int))
print(f"test: precision={p:.3f} recall={r:.3f} F1={f1:.3f}")

config = {"model_dir": MODEL_DIR, "weight": round(float(best_w), 1), "threshold": round(float(best_t), 2)}
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)
print("saved", CONFIG_PATH, config)
