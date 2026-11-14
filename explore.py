import pandas as pd

DATA = "data/deepmatcher/"

table_a = pd.read_csv(DATA + "tableA.csv")   # Abt catalog
table_b = pd.read_csv(DATA + "tableB.csv")   # Buy catalog
train = pd.read_csv(DATA + "train.csv")      # train pairs
test = pd.read_csv(DATA + "test.csv")        # test pairs
valid = pd.read_csv(DATA + "valid.csv")      # validation pairs

print("Abt products:", len(table_a))
print("Buy products:", len(table_b))
print("Train pairs:", len(train))
print("Test pairs:", len(test))
print("Validation pairs:", len(valid))


print(train.head())
print(train["label"].value_counts())


def attach_text(pairs):
    """Add the product names from both catalogs to each (ltable_id, rtable_id) pair."""
    left = table_a[["id", "name"]].rename(columns={"id": "ltable_id", "name": "left_name"})
    right = table_b[["id", "name"]].rename(columns={"id": "rtable_id", "name": "right_name"})
    return pairs.merge(left, on="ltable_id").merge(right, on="rtable_id")


train_text = attach_text(train)

pd.set_option("display.max_colwidth", 70)
print("\n--- matches (label = 1) ---")
print(train_text[train_text["label"] == 1][["left_name", "right_name"]].head(3))
print("\n--- non-matches (label = 0) ---")
print(train_text[train_text["label"] == 0][["left_name", "right_name"]].head(3))

# Leakage check: how many products appear in both train and test?
shared_a = set(train["ltable_id"]) & set(test["ltable_id"])
shared_b = set(train["rtable_id"]) & set(test["rtable_id"])
print("\nAbt products in both train and test:", len(shared_a), "of", test["ltable_id"].nunique())
print("Buy products in both train and test:", len(shared_b), "of", test["rtable_id"].nunique())
