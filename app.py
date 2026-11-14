"""Product Matching API: is this the same product?"""
import json

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from data import load_catalogs
from matchers import EmbeddingMatcher, HybridMatcher, TfidfMatcher
from search import CatalogSearch, embedding_vectors, hybrid_vectors, tfidf_vectors

CONFIG_PATH = "models/hybrid_config.json"   # written by tune_weight.py

app = FastAPI(title="Product Matching API")

# Built once, when the server starts.
with open(CONFIG_PATH) as f:
    config = json.load(f)
THRESHOLD = config["threshold"]

table_a, table_b = load_catalogs()
matcher = HybridMatcher(TfidfMatcher(), EmbeddingMatcher(config["model_dir"]), weight=config["weight"])
matcher.fit(list(table_a["name"]) + list(table_b["name"]))   # TF-IDF needs its vocabulary

# FAISS index over the Buy catalog, using the SAME hybrid score as /match (same models, same weight).
catalog = CatalogSearch(
    hybrid_vectors(tfidf_vectors(matcher.first.vectorizer), embedding_vectors(matcher.second.model), config["weight"]),
    table_b["name"],
)


class MatchRequest(BaseModel):
    left: str    # product title from store A
    right: str   # product title from store B


class MatchResponse(BaseModel):
    score: float         # similarity between 0 and 1
    same_product: bool   # score > THRESHOLD


@app.get("/", include_in_schema=False)
def root():
    """Send visitors of the bare address to the interactive docs."""
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    """Is the service up?"""
    return {"status": "ok"}


@app.post("/match", response_model=MatchResponse)
def match(request: MatchRequest):
    """Score two product titles and decide whether they are the same product."""
    score = float(matcher.score([request.left], [request.right])[0])
    return MatchResponse(score=score, same_product=score > THRESHOLD)


class SearchRequest(BaseModel):
    title: str
    k: int = Field(5, ge=1, le=20)


class SearchResult(BaseModel):
    id: int
    name: str
    score: float
    same_product: bool


@app.post("/search", response_model=list[SearchResult])
def search(request: SearchRequest):
    """Find the k catalog products most similar to one title."""
    results = []
    for row, score in catalog.search(request.title, request.k):
        results.append(SearchResult(
            id=int(table_b["id"][row]),
            name=table_b["name"][row],
            score=score,
            same_product=score > THRESHOLD,
        ))
    return results
