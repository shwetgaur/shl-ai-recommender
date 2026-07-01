"""Hybrid retrieval over the catalog: BM25 lexical + optional dense embeddings.

Design goals:
- Always work. If sentence-transformers cannot be imported or the model cannot
  load (e.g. constrained memory / offline), we transparently fall back to
  BM25-only lexical retrieval. The service never crashes because of retrieval.
- Be cheap at request time. Catalog embeddings are computed once and cached to
  disk; only the (short) query is embedded per request.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .catalog import Assessment, Catalog
from .config import ROOT_DIR, Settings, get_settings

logger = logging.getLogger(__name__)

# Keep alphanumerics plus a few symbols meaningful in tech skills (c++, c#, .net).
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with", "we",
    "our", "need", "want", "is", "are", "be", "at", "as", "by", "this", "that",
    "i", "you", "it", "they", "them", "who", "what", "which", "should", "would",
    "can", "could", "will", "do", "does", "have", "has", "help", "please",
    "hiring", "hire", "role", "assessment", "assessments", "test", "tests",
}


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall((text or "").lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


class SemanticIndex:
    """Dense-embedding index. Disabled (enabled=False) if the model won't load."""

    def __init__(self, catalog: Catalog, settings: Settings):
        self.enabled = False
        self._model = None
        self._matrix: np.ndarray | None = None
        try:
            self._build(catalog, settings)
            self.enabled = True
            logger.info("Semantic index ready (%s).", settings.embedding_model)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Semantic index unavailable, using lexical only: %s", exc)

    def _cache_path(self, settings: Settings, fingerprint: str) -> Path:
        cache_dir = ROOT_DIR / "data" / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe = settings.embedding_model.replace("/", "_")
        return cache_dir / f"emb_{safe}_{fingerprint}.npy"

    def _build(self, catalog: Catalog, settings: Settings) -> None:
        from sentence_transformers import SentenceTransformer

        docs = [self._doc(a) for a in catalog.assessments]
        fingerprint = hashlib.sha1(
            (settings.embedding_model + "|" + "|".join(a.url for a in catalog.assessments)
             ).encode("utf-8")
        ).hexdigest()[:12]
        cache = self._cache_path(settings, fingerprint)

        self._model = SentenceTransformer(settings.embedding_model)
        if cache.exists():
            self._matrix = np.load(cache)
            if self._matrix.shape[0] == len(docs):
                return
        matrix = self._model.encode(
            docs, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        self._matrix = np.asarray(matrix, dtype=np.float32)
        np.save(cache, self._matrix)

    @staticmethod
    def _doc(a: Assessment) -> str:
        return f"{a.name}. Type: {', '.join(a.keys)}. {a.description}"

    def scores(self, query: str) -> np.ndarray:
        if not self.enabled or self._model is None or self._matrix is None:
            return np.array([])
        q = self._model.encode([query], normalize_embeddings=True)
        return (self._matrix @ np.asarray(q, dtype=np.float32).T).ravel()


class HybridRetriever:
    def __init__(self, catalog: Catalog, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.catalog = catalog
        self._corpus = [tokenize(a.search_text) for a in catalog.assessments]
        # Avoid empty token lists breaking BM25.
        self._corpus = [toks if toks else ["_"] for toks in self._corpus]
        self._bm25 = BM25Okapi(self._corpus)

        mode = self.settings.retrieval_mode.lower()
        if mode == "lexical":
            self._semantic = None
            self.mode = "lexical"
        else:
            self._semantic = SemanticIndex(catalog, self.settings)
            self.mode = "hybrid" if self._semantic.enabled else "lexical"

    def lexical_scores(self, query: str) -> np.ndarray:
        toks = tokenize(query) or ["_"]
        return np.asarray(self._bm25.get_scores(toks), dtype=np.float32)

    def search(self, query: str, top_k: int | None = None) -> list[tuple[Assessment, float]]:
        top_k = top_k or self.settings.retrieval_top_k
        query = (query or "").strip()
        if not query:
            return []

        lex = _minmax(self.lexical_scores(query))
        if self._semantic and self._semantic.enabled:
            sem = self._semantic.scores(query)
            sem = _minmax(sem) if sem.size else np.zeros_like(lex)
            w = float(self.settings.hybrid_semantic_weight)
            combined = w * sem + (1.0 - w) * lex
        else:
            combined = lex

        n = min(top_k, len(self.catalog.assessments))
        # argpartition for speed, then sort the top slice.
        idx = np.argpartition(-combined, n - 1)[:n]
        idx = idx[np.argsort(-combined[idx])]
        return [(self.catalog.assessments[i], float(combined[i])) for i in idx]
