"""Load, normalize and index the SHL product catalog.

The catalog is the single source of truth. Every recommendation the agent
returns is resolved back to a catalog entry here, so the agent can never
surface a name or URL that is not grounded in the catalog.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

# SHL's canonical test-type single-letter codes, derived from the human-readable
# `keys` field. This mapping matches the codes used in the provided sample traces.
KEY_TO_LETTER: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

_LETTER_ORDER = ["A", "B", "C", "D", "E", "K", "P", "S"]


def _norm_url(url: str) -> str:
    """Normalize a URL for equality comparisons (lowercase, no trailing slash)."""
    return (url or "").strip().lower().rstrip("/")


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


@dataclass
class Assessment:
    entity_id: str
    name: str
    url: str
    description: str
    keys: list[str]
    test_type: str
    job_levels: list[str]
    languages: list[str]
    duration: str
    remote: str
    adaptive: str
    # Precomputed lowercase blob used for lexical retrieval.
    search_text: str = field(default="", repr=False)

    def to_recommendation(self) -> dict:
        return {"name": self.name, "url": self.url, "test_type": self.test_type}

    def brief(self) -> dict:
        """Compact representation handed to the LLM as grounding context."""
        return {
            "id": self.entity_id,
            "name": self.name,
            "test_type": self.test_type,
            "keys": ", ".join(self.keys),
            "duration": self.duration or "-",
            "job_levels": ", ".join(self.job_levels) if self.job_levels else "-",
            "languages": self._languages_display(),
            "remote": self.remote or "-",
            "adaptive": self.adaptive or "-",
            "description": _truncate(self.description, 200),
            "url": self.url,
        }

    def _languages_display(self, head: int = 4) -> str:
        if not self.languages:
            return "-"
        if len(self.languages) <= head:
            return ", ".join(self.languages)
        return f"{', '.join(self.languages[:head])} (+{len(self.languages) - head} more)"


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _letters_from_keys(keys: Iterable[str]) -> str:
    letters: list[str] = []
    for k in keys:
        letter = KEY_TO_LETTER.get(k.strip())
        if letter and letter not in letters:
            letters.append(letter)
    letters.sort(key=lambda x: _LETTER_ORDER.index(x) if x in _LETTER_ORDER else 99)
    return ",".join(letters)


class Catalog:
    """In-memory catalog with id/url/name indexes and search blobs."""

    def __init__(self, assessments: list[Assessment]):
        self.assessments = assessments
        self._by_id = {a.entity_id: a for a in assessments}
        self._by_url = {_norm_url(a.url): a for a in assessments}
        self._by_name = {_norm_name(a.name): a for a in assessments}

    def __len__(self) -> int:
        return len(self.assessments)

    def get(self, entity_id: str) -> Assessment | None:
        return self._by_id.get(str(entity_id).strip())

    def by_url(self, url: str) -> Assessment | None:
        return self._by_url.get(_norm_url(url))

    def by_name(self, name: str) -> Assessment | None:
        return self._by_name.get(_norm_name(name))

    def resolve(self, ref: str) -> Assessment | None:
        """Resolve an id, URL, or name to an assessment (best effort)."""
        if not ref:
            return None
        ref = str(ref).strip()
        return self.get(ref) or self.by_url(ref) or self.by_name(ref)


def _build_search_text(entry: dict, name: str, keys: list[str]) -> str:
    parts = [
        name,
        name,  # weight the name a little by repeating it
        entry.get("description", "") or "",
        " ".join(keys),
        " ".join(entry.get("job_levels", []) or []),
        " ".join(entry.get("languages", []) or []),
    ]
    return re.sub(r"\s+", " ", " ".join(parts)).strip().lower()


def _normalize_entry(entry: dict) -> Assessment | None:
    name = (entry.get("name") or "").strip()
    url = (entry.get("link") or entry.get("url") or "").strip()
    if not name or not url:
        return None
    keys = [k for k in (entry.get("keys") or []) if isinstance(k, str) and k.strip()]
    return Assessment(
        entity_id=str(entry.get("entity_id") or url),
        name=name,
        url=url,
        description=entry.get("description", "") or "",
        keys=keys,
        test_type=_letters_from_keys(keys),
        job_levels=[j for j in (entry.get("job_levels") or []) if isinstance(j, str)],
        languages=[l for l in (entry.get("languages") or []) if isinstance(l, str)],
        duration=(entry.get("duration") or "").strip(),
        remote=(entry.get("remote") or "").strip(),
        adaptive=(entry.get("adaptive") or "").strip(),
        search_text=_build_search_text(entry, name, keys),
    )


def _load_raw(settings: Settings) -> list[dict]:
    path: Path = settings.catalog_file
    if path.exists():
        text = path.read_text(encoding="utf-8")
        # Guard against files that were saved with a header preamble.
        idx = text.find("[")
        if idx > 0:
            text = text[idx:]
        # strict=False tolerates raw control characters inside strings, which
        # the source catalog occasionally contains.
        return json.loads(text, strict=False)

    # Fallback: fetch from the published URL and cache to disk.
    logger.warning("Catalog file %s missing; fetching from %s", path, settings.catalog_url)
    import requests

    resp = requests.get(settings.catalog_url, timeout=30)
    resp.raise_for_status()
    data = json.loads(resp.text, strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def load_catalog(settings: Settings | None = None) -> Catalog:
    settings = settings or get_settings()
    raw = _load_raw(settings)
    assessments: list[Assessment] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        a = _normalize_entry(entry)
        if a is None:
            continue
        key = _norm_url(a.url)
        if key in seen:  # de-duplicate on URL
            continue
        seen.add(key)
        assessments.append(a)
    if not assessments:
        raise RuntimeError("Catalog loaded but contained zero valid assessments.")
    logger.info("Loaded %d assessments from catalog.", len(assessments))
    return Catalog(assessments)
