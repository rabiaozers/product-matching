"""Load the Abt-Buy dataset and attach product names to the labelled pairs."""
import pandas as pd

DATA = "data/deepmatcher/"


def load_catalogs():
    """Return the two product catalogs: Abt (table A) and Buy (table B)."""
    table_a = pd.read_csv(DATA + "tableA.csv")
    table_b = pd.read_csv(DATA + "tableB.csv")
    return table_a, table_b


def load_pairs(split, table_a, table_b):
    """Load train / valid / test pairs and add left_name and right_name columns."""
    pairs = pd.read_csv(DATA + split + ".csv")
    left = table_a[["id", "name"]].rename(columns={"id": "ltable_id", "name": "left_name"})
    right = table_b[["id", "name"]].rename(columns={"id": "rtable_id", "name": "right_name"})
    return pairs.merge(left, on="ltable_id").merge(right, on="rtable_id")
