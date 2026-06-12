"""
Semantic paper vector store.
Primary  : sentence-transformers (all-MiniLM-L6-v2) — cosine similarity
Fallback : custom TF-IDF — when sentence-transformers is unavailable
"""

import re
import math
import json
import os
from collections import Counter

import numpy as np

# ── lazy embedding model ──────────────────────────────────────────────────────

_embed_model = None
_embed_ready = None          # None = untested, True/False = outcome


def _load_embed_model():
    """Load sentence-transformers once; cache result in module globals."""
    global _embed_model, _embed_ready
    if _embed_ready is not None:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        _embed_ready = True
        print("[STORE] Semantic search ready — all-MiniLM-L6-v2")
    except Exception as exc:
        _embed_ready = False
        print(f"[STORE] Falling back to TF-IDF ({exc})")
    return _embed_model


# ── TF-IDF helpers ────────────────────────────────────────────────────────────

_STOPWORDS = {
    "a","an","the","and","or","of","in","to","for","with","on","at","by",
    "is","are","was","were","be","been","have","has","had","this","that",
    "from","as","but","not","it","its","which","who","may","also","can",
    "all","one","more","than","these","those","our","their","we","they",
    "he","she","clinical","study","patients","patient","data","results",
    "conclusion","background","methods","objective","aims","purpose",
    "use","used","using","new","among","within","between","during",
    "after","before","without","based","compared","associated",
    "significantly","including","showed","found","however","although",
}


def _tokenize(text: str) -> list:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _tfidf_scores(query_tokens: list, all_doc_tokens: list) -> list:
    N  = len(all_doc_tokens)
    df = Counter()
    for doc in all_doc_tokens:
        for t in set(doc):
            df[t] += 1
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in df}
    results = []
    for doc in all_doc_tokens:
        counts = Counter(doc)
        length = max(len(doc), 1)
        sc = sum((counts[t] / length) * idf.get(t, 1.0) for t in query_tokens if t in counts)
        results.append(round(sc, 6))
    return results


# ── PaperVectorStore ──────────────────────────────────────────────────────────

class PaperVectorStore:
    """
    Hybrid paper store with two search modes:
    - Semantic  : dense cosine similarity via sentence-transformers
    - TF-IDF    : sparse bag-of-words fallback (no ML deps)

    Both modes attach a 'score' field to returned papers.
    Papers are de-duplicated by URL and persisted to disk.
    """

    def __init__(self, cache_path: str = None):
        self._papers     = []
        self._embeddings = None          # np.ndarray (n, d) or None when stale
        self._cache_path = cache_path
        self._emb_path   = (
            cache_path.replace(".json", "_embeddings.npy") if cache_path else None
        )
        self._load_from_disk()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load_from_disk(self):
        if self._cache_path and os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    self._papers = json.load(f)
            except Exception:
                self._papers = []

        if self._emb_path and os.path.exists(self._emb_path) and self._papers:
            try:
                embs = np.load(self._emb_path)
                if embs.shape[0] == len(self._papers):
                    self._embeddings = embs
            except Exception:
                self._embeddings = None

    def _save_json(self):
        if not self._cache_path:
            return
        try:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._papers, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _save_embeddings(self):
        if (
            self._emb_path is not None
            and self._embeddings is not None
            and self._embeddings.shape[0] == len(self._papers)
        ):
            try:
                np.save(self._emb_path, self._embeddings)
            except Exception:
                pass

    # ── embedding helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _paper_text(p: dict) -> str:
        return f"{p.get('title','')} {p.get('abstract','')} {p.get('journal','')}"

    def _embeddings_fresh(self) -> bool:
        return (
            self._embeddings is not None
            and self._embeddings.shape[0] == len(self._papers)
            and len(self._papers) > 0
        )

    def _refresh_embeddings(self):
        """Encode all papers; persists result to disk."""
        model = _load_embed_model()
        if not _embed_ready or model is None:
            return
        texts = [self._paper_text(p) for p in self._papers]
        self._embeddings = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        self._save_embeddings()

    # ── public API ────────────────────────────────────────────────────────────

    def add_papers(self, papers: list) -> int:
        """Add papers (de-duplicated by link URL). Returns count added."""
        existing = {p.get("link") for p in self._papers}
        new = [p for p in papers if p.get("link") and p["link"] not in existing]
        if not new:
            return 0
        self._papers.extend(new)
        self._embeddings = None          # mark stale — refresh on next search
        self._save_json()
        return len(new)

    def search(self, query: str, top_k: int = 12) -> list:
        """
        Return top_k most relevant papers for query.
        Each paper dict has an added 'score' key (cosine sim or TF-IDF score).
        """
        if not self._papers:
            return []

        if not self._embeddings_fresh():
            self._refresh_embeddings()

        if self._embeddings_fresh():
            model = _load_embed_model()
            q_emb  = model.encode([query], normalize_embeddings=True)[0]
            scores = (self._embeddings @ q_emb).tolist()
            ranked = sorted(zip(scores, self._papers), key=lambda x: x[0], reverse=True)
            return [{**p, "score": round(s, 4)} for s, p in ranked[:top_k]]

        # TF-IDF fallback
        q_tokens = _tokenize(query)
        all_toks = [_tokenize(self._paper_text(p)) for p in self._papers]
        if q_tokens:
            scores = _tfidf_scores(q_tokens, all_toks)
        else:
            scores = [0.0] * len(self._papers)
        ranked = sorted(zip(scores, self._papers), key=lambda x: x[0], reverse=True)
        return [{**p, "score": round(s, 4)} for s, p in ranked[:top_k]]

    def clear(self):
        self._papers     = []
        self._embeddings = None
        for path in filter(None, [self._cache_path, self._emb_path]):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    @property
    def count(self) -> int:
        return len(self._papers)

    @property
    def using_semantic(self) -> bool:
        return bool(_embed_ready)


# ── module-level singleton ────────────────────────────────────────────────────

_store: PaperVectorStore = None


def get_store() -> PaperVectorStore:
    global _store
    if _store is None:
        cache_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".paper_cache.json"
        )
        _store = PaperVectorStore(cache_path=cache_file)
    return _store
