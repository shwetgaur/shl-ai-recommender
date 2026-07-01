"""Validate the LLM policy path using a scripted fake LLM (no network)."""
from __future__ import annotations

import pytest

from app.agent import Agent
from app.catalog import load_catalog
from app.config import get_settings
from app.retrieval import HybridRetriever
from app.schemas import Message


class FakeLLM:
    """Minimal stand-in for LLMClient returning scripted JSON objects."""

    def __init__(self, script):
        self._script = script
        self.calls = 0

    def available_providers(self):
        return ["fake"]

    @property
    def has_provider(self):
        return True

    def generate_json(self, system, user, **kwargs):
        obj = self._script(user) if callable(self._script) else self._script
        self.calls += 1
        return obj


@pytest.fixture(scope="module")
def base():
    settings = get_settings()
    catalog = load_catalog(settings)
    retriever = HybridRetriever(catalog, settings)
    return settings, catalog, retriever


def _first_two_ids(catalog):
    return [catalog.assessments[0].entity_id, catalog.assessments[1].entity_id]


def test_llm_recommend_grounds_ids(base):
    settings, catalog, retriever = base
    ids = _first_two_ids(catalog)
    fake = FakeLLM({
        "action": "recommend",
        "reply": "Here you go.",
        "recommendation_ids": ids,
        "end_of_conversation": False,
    })
    agent = Agent(catalog, retriever, fake, settings)
    resp = agent.handle([Message(role="user", content="Hiring a Java backend engineer with Spring.")])
    got = {r.url for r in resp.recommendations}
    expected = {catalog.get(i).url for i in ids}
    assert got == expected


def test_llm_invalid_ids_are_dropped_then_grounded(base):
    settings, catalog, retriever = base
    fake = FakeLLM({
        "action": "recommend",
        "reply": "Here.",
        "recommendation_ids": ["not-a-real-id", "9999999"],
        "end_of_conversation": False,
    })
    agent = Agent(catalog, retriever, fake, settings)
    resp = agent.handle([Message(role="user", content="Data analyst with SQL and statistics.")])
    # Invalid ids dropped, but agent still grounds from retrieval rather than hallucinate.
    catalog_urls = {a.url for a in catalog.assessments}
    assert all(r.url in catalog_urls for r in resp.recommendations)
    assert len(resp.recommendations) >= 1


def test_llm_vague_opening_forced_to_clarify(base):
    settings, catalog, retriever = base
    ids = _first_two_ids(catalog)
    # Even if the LLM tries to recommend on a vague opener, the guard blocks it.
    fake = FakeLLM({
        "action": "recommend",
        "reply": "Recommending anyway.",
        "recommendation_ids": ids,
        "end_of_conversation": False,
    })
    agent = Agent(catalog, retriever, fake, settings)
    resp = agent.handle([Message(role="user", content="I need an assessment.")])
    assert resp.recommendations == []


def test_llm_clarify_has_no_recommendations(base):
    settings, catalog, retriever = base
    fake = FakeLLM({
        "action": "clarify",
        "reply": "What seniority?",
        "recommendation_ids": [],
        "end_of_conversation": False,
    })
    agent = Agent(catalog, retriever, fake, settings)
    resp = agent.handle([Message(role="user", content="Hiring a software engineer.")])
    assert resp.recommendations == []
    assert resp.end_of_conversation is False


def test_llm_max_ten_enforced(base):
    settings, catalog, retriever = base
    many = [a.entity_id for a in catalog.assessments[:20]]
    fake = FakeLLM({
        "action": "recommend",
        "reply": "Big list.",
        "recommendation_ids": many,
        "end_of_conversation": True,
    })
    agent = Agent(catalog, retriever, fake, settings)
    resp = agent.handle([Message(role="user", content="Full battery for graduate scheme, cognitive personality sjt.")])
    assert len(resp.recommendations) <= 10


def test_llm_confirmation_can_end(base):
    settings, catalog, retriever = base
    ids = _first_two_ids(catalog)
    fake = FakeLLM({
        "action": "recommend",
        "reply": "Confirmed.",
        "recommendation_ids": ids,
        "end_of_conversation": True,
    })
    agent = Agent(catalog, retriever, fake, settings)
    convo = [
        Message(role="user", content="Hiring plant operators, safety critical."),
        Message(role="assistant", content="Options..."),
        Message(role="user", content="Perfect, confirmed."),
    ]
    resp = agent.handle(convo)
    assert resp.end_of_conversation is True
    assert 1 <= len(resp.recommendations) <= 10


def test_llm_exception_falls_back(base):
    settings, catalog, retriever = base

    class BoomLLM(FakeLLM):
        def generate_json(self, system, user, **kwargs):
            raise RuntimeError("provider down")

    agent = Agent(catalog, retriever, BoomLLM({}), settings)
    resp = agent.handle([Message(role="user", content="Java backend engineer, Spring, SQL. Confirmed.")])
    # Falls back deterministically to a grounded shortlist.
    catalog_urls = {a.url for a in catalog.assessments}
    assert all(r.url in catalog_urls for r in resp.recommendations)
