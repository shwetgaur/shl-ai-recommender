import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force deterministic (no-LLM) behavior and lexical retrieval for fast, hermetic
# tests that don't depend on network, API keys, or model downloads.
os.environ.setdefault("RETRIEVAL_MODE", "lexical")
os.environ.setdefault("LLM_PROVIDER_PRIORITY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("OPENAI_API_KEY", "")

import pytest  # noqa: E402

from app.agent import Agent  # noqa: E402
from app.catalog import load_catalog  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.retrieval import HybridRetriever  # noqa: E402


@pytest.fixture(scope="session")
def catalog():
    return load_catalog(get_settings())


@pytest.fixture(scope="session")
def agent(catalog):
    settings = get_settings()
    retriever = HybridRetriever(catalog, settings)
    return Agent(catalog, retriever, LLMClient(settings), settings)
