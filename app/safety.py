"""Scope and safety guardrails.

Two layers protect the agent:
1. Deterministic heuristics here catch the clearest prompt-injection and
   obviously off-topic requests even when no LLM is available.
2. The LLM policy (see agent/prompts) makes the nuanced in-scope decisions.

We stay conservative: heuristics only fire on high-precision patterns so we
never refuse a legitimate assessment query.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts|messages|context)",
    r"disregard (all |the )?(previous|prior|above|your) (instructions|rules|prompt)",
    r"forget (all |everything |your )?(previous|prior|the above|instructions)",
    r"(reveal|show|print|repeat|expose|leak|tell me) (me )?(your )?(the )?(system )?(prompt|instructions|rules)",
    r"what (are|were) your (system )?(instructions|prompt|rules)",
    r"repeat (the words|everything) above",
    r"you are now\b",
    r"act as (?!a hiring|an? recruiter)",  # 'act as DAN', etc.
    r"\bdeveloper mode\b",
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"pretend (you are|to be)\b",
    r"override your (rules|instructions|guardrails)",
    r"new (system )?prompt\s*:",
    r"</?(system|instructions)>",
]

# High-precision off-topic markers (deliberately narrow).
_OFF_TOPIC_PATTERNS = [
    r"\bweather\b",
    r"\b(write|generate|debug|fix) (me )?(a |some )?(code|python|java(script)?|sql|program|script|function|poem|essay|story|email)\b",
    r"\b(tell|say) (me )?a joke\b",
    r"\bwho (won|is the president|is the ceo)\b",
    r"\b(recipe|cook|football|movie|song|lyrics)\b",
    r"\bstock (price|market)\b",
    r"\btranslate\b.*\b(into|to)\b",
    r"\bwhat is the capital of\b",
]

# Requests for advice we explicitly refuse (legal / regulatory / general HR).
_LEGAL_PATTERNS = [
    r"\b(legal(ly)?|lawsuit|lawful|liabilit|discriminat|gdpr|eeoc|adverse impact|comply|compliance|regulation|regulator|statute|hipaa (require|law))\b.*\b(require|allowed|permit|legal|sue|obligat|must)\b",
    r"\bare we (legally )?(required|allowed|permitted|obligated)\b",
    r"\bis it (legal|lawful|illegal)\b",
    r"\bcan (i|we) get sued\b",
]

_injection_re = [re.compile(p, re.I) for p in _INJECTION_PATTERNS]
_offtopic_re = [re.compile(p, re.I) for p in _OFF_TOPIC_PATTERNS]
_legal_re = [re.compile(p, re.I) for p in _LEGAL_PATTERNS]


@dataclass
class SafetyVerdict:
    kind: str  # "ok" | "injection" | "off_topic" | "legal"
    reason: str = ""

    @property
    def is_refusal(self) -> bool:
        return self.kind in {"injection", "off_topic", "legal"}


def check_injection(text: str) -> bool:
    return any(r.search(text or "") for r in _injection_re)


def check_off_topic(text: str) -> bool:
    return any(r.search(text or "") for r in _offtopic_re)


def check_legal(text: str) -> bool:
    return any(r.search(text or "") for r in _legal_re)


def screen_latest_user_message(text: str) -> SafetyVerdict:
    """Fast deterministic screen of the most recent user message."""
    text = text or ""
    if check_injection(text):
        return SafetyVerdict("injection", "prompt-injection pattern detected")
    if check_legal(text):
        return SafetyVerdict("legal", "legal/regulatory advice request")
    if check_off_topic(text):
        return SafetyVerdict("off_topic", "request unrelated to SHL assessments")
    return SafetyVerdict("ok")


REFUSAL_MESSAGES = {
    "injection": (
        "I can only help with selecting SHL assessments from the catalog. "
        "I can't change my instructions or step outside that scope. "
        "Tell me about the role you're hiring for and I'll suggest suitable assessments."
    ),
    "off_topic": (
        "I'm focused on helping you choose SHL assessments for hiring. "
        "That question is outside what I can help with. "
        "If you describe the role or skills you're assessing, I can recommend options from the SHL catalog."
    ),
    "legal": (
        "That's a legal/compliance question I'm not able to advise on - your legal or "
        "compliance team is the right resource. I can, however, help you select SHL "
        "assessments that measure the skills or behaviours you care about."
    ),
}
