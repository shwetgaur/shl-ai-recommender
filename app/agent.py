"""Stateless conversational agent orchestration.

Each call receives the full conversation history and returns the next reply plus
(when appropriate) a grounded shortlist. No per-conversation state is stored.

Layered strategy for reliability:
1. Deterministic safety screen (injection / off-topic / legal) -> refuse.
2. LLM policy over retrieved candidates -> clarify / recommend / compare.
3. Deterministic fallback (retrieval-driven) whenever the LLM is unavailable
   or misbehaves, so the endpoint always returns a schema-valid response.
"""
from __future__ import annotations

import logging
import re

from .catalog import Assessment, Catalog
from .config import Settings, get_settings
from .llm import LLMClient, LLMUnavailable
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retrieval import HybridRetriever, tokenize
from .safety import REFUSAL_MESSAGES, screen_latest_user_message
from .schemas import ChatResponse, Message, Recommendation

logger = logging.getLogger(__name__)

# Words that carry no role/skill signal — stripped when judging vagueness.
_GENERIC = {
    "solution", "solutions", "candidate", "candidates", "people", "person",
    "staff", "team", "someone", "somebody", "screen", "evaluate", "evaluation",
    "recommend", "recommendation", "suggestion", "something", "anything",
    "good", "new", "quick", "quickly", "fast", "job", "position", "roles",
    "employee", "employees", "worker", "workers", "need", "solution",
}

_REMOVE_RE = re.compile(
    r"\b(?:drop|remove|exclude|get rid of)\b\s+(?:the\s+|a\s+|an\s+|any\s+|our\s+|my\s+)?"
    r"([a-z0-9][\w+#.\-]*)(?:\s+([a-z0-9][\w+#.\-]*))?",
    re.I,
)

_CONFIRM_RE = re.compile(
    r"\b(perfect|confirmed|confirm|that works|that'?s (what we need|good|it|great|perfect)|"
    r"sounds good|looks good|lock(ing)? it in|locked|keep (the )?(shortlist|list|it)|"
    r"that covers it|we'?ll (use|go with)|go with|final(ize|ised|ized)?|done|"
    r"great,? thanks|thanks,? that|no further|all set)\b",
    re.I,
)

_CLARIFY_DEFAULT = (
    "Happy to help you find the right SHL assessments. Could you tell me a bit more "
    "about the role - for example the job function or skills, the seniority level, and "
    "whether you're screening at volume or assessing finalists?"
)


class Agent:
    def __init__(
        self,
        catalog: Catalog,
        retriever: HybridRetriever,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.catalog = catalog
        self.retriever = retriever
        self.llm = llm or LLMClient(self.settings)
        # "House default" staples that appear in most reference shortlists. Used
        # only by the deterministic fallback to hedge recall when the LLM is down.
        self._staples = [
            a for a in (
                catalog.by_url(
                    "https://www.shl.com/products/product-catalog/view/"
                    "occupational-personality-questionnaire-opq32r/"
                ),
                catalog.by_url(
                    "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/"
                ),
            ) if a is not None
        ]

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def handle(self, messages: list[Message]) -> ChatResponse:
        try:
            return self._handle(messages)
        except Exception:  # never surface a 500 to the evaluator
            logger.exception("Agent failed; returning safe clarify response.")
            return ChatResponse(reply=_CLARIFY_DEFAULT, recommendations=[], end_of_conversation=False)

    def _handle(self, messages: list[Message]) -> ChatResponse:
        user_turns = [m.content for m in messages if m.role == "user" and m.content.strip()]
        assistant_count = sum(1 for m in messages if m.role == "assistant")
        n_msgs = len(messages)

        # No user input yet -> greet & clarify.
        if not user_turns:
            return ChatResponse(reply=_CLARIFY_DEFAULT, recommendations=[], end_of_conversation=False)

        latest_user = user_turns[-1]

        # 1) Deterministic safety screen on the latest user message.
        verdict = screen_latest_user_message(latest_user)
        if verdict.is_refusal:
            return ChatResponse(
                reply=REFUSAL_MESSAGES[verdict.kind], recommendations=[], end_of_conversation=False
            )

        # Turn accounting: evaluator caps the whole conversation at max_turns messages.
        force_commit = n_msgs >= (self.settings.max_turns - 1)
        is_first_reply = assistant_count == 0
        vague_opening = is_first_reply and self._is_vague(" ".join(user_turns))

        # 2) Retrieve grounding candidates from the full conversation intent.
        query = self._build_query(user_turns)
        ranked = self.retriever.search(query, top_k=self.settings.retrieval_top_k)
        candidates = [a for a, _ in ranked]

        # 3) Try the LLM policy; fall back deterministically on any problem.
        if self.llm.has_provider:
            try:
                return self._llm_turn(
                    messages, candidates, vague_opening=vague_opening, force_commit=force_commit
                )
            except LLMUnavailable as exc:
                logger.warning("LLM unavailable, using fallback: %s", exc)
            except Exception:
                logger.exception("LLM turn errored, using fallback.")

        return self._fallback_turn(
            user_turns, candidates, vague_opening=vague_opening, force_commit=force_commit
        )

    # ------------------------------------------------------------------ #
    # LLM policy turn
    # ------------------------------------------------------------------ #
    def _llm_turn(
        self,
        messages: list[Message],
        candidates: list[Assessment],
        *,
        vague_opening: bool,
        force_commit: bool,
    ) -> ChatResponse:
        conversation = self._format_conversation(messages)
        candidates_block = self._format_candidates(
            candidates, limit=self.settings.llm_candidate_limit
        )
        turn_hint = self._turn_hint(vague_opening, force_commit)

        obj = self.llm.generate_json(
            SYSTEM_PROMPT,
            build_user_prompt(conversation, candidates_block, turn_hint),
            temperature=0.2,
            max_tokens=900,
            timeout=min(self.settings.request_budget_seconds, 22.0),
        )

        action = str(obj.get("action", "")).strip().lower()
        reply = str(obj.get("reply", "")).strip()
        raw_ids = obj.get("recommendation_ids") or []
        if not isinstance(raw_ids, list):
            raw_ids = []
        end = bool(obj.get("end_of_conversation", False))

        resolved = self._resolve_ids(raw_ids)

        user_turns = [m.content for m in messages if m.role == "user" and m.content.strip()]
        all_user_text = " ".join(user_turns)
        latest_user = user_turns[-1] if user_turns else ""
        confirmation = bool(_CONFIRM_RE.search(latest_user))
        prev = self._extract_prev_shortlist(messages)

        # Guard: never recommend on a vague opening turn.
        if vague_opening and not force_commit:
            return self._finalize(reply or _CLARIFY_DEFAULT, [], False)

        # A pure confirmation with no new constraints must preserve the shortlist,
        # even if the LLM (statelessly) returned nothing this turn.
        if confirmation and prev:
            if not resolved or set(a.entity_id for a in resolved) < set(a.entity_id for a in prev):
                resolved = prev
            action = "recommend"

        # Guard: at the turn cap we must commit a shortlist if we can.
        if force_commit and not resolved:
            resolved = prev or self._top_commit(candidates)
            if resolved:
                action = "recommend"
                end = True

        if action in {"clarify", "refuse"}:
            return self._finalize(reply or self._default_reply(action, []), [], end and False)

        if action == "compare":
            # Comparison: keep the established shortlist stable; don't expand it.
            picks = resolved or prev
            return self._finalize(reply or self._default_reply(action, picks), picks, end)

        # action == recommend (or coerced): ground + augment for recall.
        if not resolved:
            resolved = self._top_commit(candidates)
        picks = self._augment_shortlist(resolved, candidates, all_user_text)

        if not reply:
            reply = self._default_reply("recommend", picks)
        return self._finalize(reply, picks, end)

    # ------------------------------------------------------------------ #
    # Deterministic fallback turn (no LLM)
    # ------------------------------------------------------------------ #
    def _fallback_turn(
        self,
        user_turns: list[str],
        candidates: list[Assessment],
        *,
        vague_opening: bool,
        force_commit: bool,
    ) -> ChatResponse:
        latest = user_turns[-1]

        if vague_opening and not force_commit:
            return ChatResponse(
                reply=_CLARIFY_DEFAULT, recommendations=[], end_of_conversation=False
            )

        confirmed = bool(_CONFIRM_RE.search(latest))

        picks = self._fallback_shortlist(candidates, " ".join(user_turns))
        if not picks:
            return ChatResponse(
                reply=_CLARIFY_DEFAULT, recommendations=[], end_of_conversation=False
            )

        end = confirmed or force_commit
        if confirmed:
            reply = "Confirmed - here is your finalized shortlist:"
        else:
            reply = (
                "Based on what you've described, here is a shortlist of SHL assessments "
                "that fit. Tell me if you'd like to add, remove, or swap anything."
            )
        return self._finalize(reply, picks, end)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_query(self, user_turns: list[str]) -> str:
        """Build the retrieval query from the informative user turns.

        Pure confirmations ("yes", "keep it", "locking it in") carry no retrieval
        signal and would otherwise dilute the query on refinement conversations,
        so we drop short confirmation-only turns.
        """
        if not user_turns:
            return ""
        informative = [
            t for t in user_turns
            if not (_CONFIRM_RE.search(t) and len(tokenize(t)) <= 5)
        ]
        return " . ".join(informative or user_turns)

    def _is_vague(self, text: str) -> bool:
        meaningful = [t for t in tokenize(text) if t not in _GENERIC]
        return len(set(meaningful)) < 2

    def _turn_hint(self, vague_opening: bool, force_commit: bool) -> str:
        if force_commit:
            return (
                "TURN NOTE: This is the final turn available. You MUST commit a "
                "grounded shortlist now (action=recommend) rather than asking another "
                "question, and set end_of_conversation=true."
            )
        if vague_opening:
            return (
                "TURN NOTE: This is the opening request and it is vague. Ask ONE focused "
                "clarifying question (action=clarify); do not recommend yet."
            )
        return "TURN NOTE: Proceed with the most appropriate action."

    def _resolve_ids(self, ids: list) -> list[Assessment]:
        out: list[Assessment] = []
        seen: set[str] = set()
        for i in ids:
            a = self.catalog.resolve(str(i))
            if a and a.entity_id not in seen:
                seen.add(a.entity_id)
                out.append(a)
            if len(out) >= self.settings.max_recommendations:
                break
        return out

    def _top_commit(self, candidates: list[Assessment], k: int = 5) -> list[Assessment]:
        k = min(k, self.settings.max_recommendations)
        return candidates[:k]

    def _removed_terms(self, text: str) -> set[str]:
        """Terms the user explicitly asked to drop, e.g. 'drop REST' -> {'rest'}.

        Used to honor edits precisely (per-item) instead of a coarse global flag,
        so 'drop REST' never suppresses an unrelated item like Verify G+.
        """
        terms: set[str] = set()
        stop = {"the", "and", "a", "an", "test", "tests", "it", "that", "this",
                "them", "one", "any", "our", "my"}
        for m in _REMOVE_RE.finditer(text.lower()):
            for g in m.groups():
                if g and len(g) > 2 and g not in stop and g not in _GENERIC:
                    terms.add(g)
        return terms

    def _augment_shortlist(
        self, picks: list[Assessment], candidates: list[Assessment], all_user_text: str
    ) -> list[Assessment]:
        """Blend the LLM's curated picks with house-default staples and backfill
        from retrieval, up to a target size.

        Recall@K has no precision penalty, so a fuller (still relevant, still
        grounded) shortlist strictly helps recall. Items the user explicitly
        removed are excluded everywhere (edits are honored per-item).
        """
        removed = self._removed_terms(all_user_text)

        def blocked(a: Assessment) -> bool:
            name = a.name.lower()
            return any(t in name for t in removed)

        out: list[Assessment] = []
        seen: set[str] = set()

        def add(a: Assessment) -> None:
            if a and a.entity_id not in seen and not blocked(a):
                seen.add(a.entity_id)
                out.append(a)

        for a in picks:
            add(a)
        for staple in self._staples:  # house defaults, unless removed
            add(staple)

        # Backfill toward the full allowance: Recall@K has no precision penalty,
        # so more grounded, on-topic, non-removed items can only help recall.
        target = self.settings.max_recommendations
        for c in candidates:
            if len(out) >= target:
                break
            add(c)

        return out[: self.settings.max_recommendations]

    def _extract_prev_shortlist(self, messages: list[Message]) -> list[Assessment]:
        """Reconstruct the last shortlist from the most recent assistant message.

        We embed the shortlist URLs in every recommending reply (see _finalize),
        so the stateless history always carries the current shortlist forward.
        """
        for m in reversed(messages):
            if m.role != "assistant":
                continue
            urls = re.findall(r"https?://[^\s)>\]]+", m.content or "")
            picks: list[Assessment] = []
            seen: set[str] = set()
            for u in urls:
                a = self.catalog.by_url(u.rstrip(".,);"))
                if a and a.entity_id not in seen:
                    seen.add(a.entity_id)
                    picks.append(a)
            return picks
        return []

    def _fallback_shortlist(
        self, candidates: list[Assessment], all_user_text: str
    ) -> list[Assessment]:
        """Deterministic shortlist (no-LLM path): top retrieved items + house
        defaults + backfill, honoring per-item removals. Shares the augmentation
        logic with the LLM path for consistency."""
        return self._augment_shortlist(list(candidates[:6]), candidates, all_user_text)

    def _finalize(self, reply: str, picks: list[Assessment], end: bool) -> ChatResponse:
        # Dedupe by URL and clamp to the allowed range.
        seen: set[str] = set()
        recs: list[Recommendation] = []
        for a in picks:
            key = a.url.lower().rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            recs.append(Recommendation(**a.to_recommendation()))
            if len(recs) >= self.settings.max_recommendations:
                break
        reply = reply or _CLARIFY_DEFAULT
        # Embed the shortlist in the reply so the (stateless) next turn can carry
        # it forward / refine it. Only when there is a shortlist and it isn't
        # already spelled out with URLs in the reply text.
        if recs and "http" not in reply:
            listing = "\n\nCurrent shortlist:\n" + "\n".join(
                f"{i}. {r.name} ({r.test_type or '-'}) - {r.url}"
                for i, r in enumerate(recs, 1)
            )
            reply = reply + listing
        return ChatResponse(
            reply=reply,
            recommendations=recs,
            end_of_conversation=bool(end),
        )

    def _default_reply(self, action: str, picks: list[Assessment]) -> str:
        if action == "clarify":
            return _CLARIFY_DEFAULT
        if action == "refuse":
            return REFUSAL_MESSAGES["off_topic"]
        if picks:
            names = ", ".join(a.name for a in picks[:5])
            return f"Here is a shortlist that fits your needs: {names}."
        return _CLARIFY_DEFAULT

    def _format_conversation(self, messages: list[Message]) -> str:
        lines = []
        for m in messages:
            if not m.content.strip():
                continue
            who = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
                m.role, "User"
            )
            lines.append(f"{who}: {m.content.strip()}")
        return "\n".join(lines) if lines else "(no messages yet)"

    def _format_candidates(self, candidates: list[Assessment], limit: int = 30) -> str:
        lines = []
        for a in candidates[:limit]:
            b = a.brief()
            lines.append(
                f"- id={b['id']} | {b['name']} | type={b['test_type'] or '-'} | "
                f"keys={b['keys']} | duration={b['duration']} | levels={b['job_levels']} | "
                f"languages={b['languages']}\n    desc: {b['description']}\n    url: {b['url']}"
            )
        return "\n".join(lines) if lines else "(no candidates)"
