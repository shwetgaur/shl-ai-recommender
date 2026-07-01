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
        candidates_block = self._format_candidates(candidates, limit=30)
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

        # Guard: never recommend on a vague opening turn.
        if vague_opening and not force_commit:
            action = "clarify"
            resolved = []

        # Guard: at the turn cap we must commit a shortlist if we can.
        if force_commit and not resolved:
            resolved = self._top_commit(candidates)
            if resolved:
                action = "recommend"
                end = True

        if action in {"clarify", "refuse"}:
            resolved = []
        elif action == "recommend" and not resolved:
            # LLM wanted to recommend but gave no valid ids -> ground from retrieval.
            resolved = self._top_commit(candidates)

        if not reply:
            reply = self._default_reply(action, resolved)

        return self._finalize(reply, resolved, end)

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

        picks = self._fallback_shortlist(candidates, " ".join(user_turns))
        if not picks:
            return ChatResponse(
                reply=_CLARIFY_DEFAULT, recommendations=[], end_of_conversation=False
            )

        confirmed = bool(_CONFIRM_RE.search(latest))
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
        # Weight the most recent message a little more heavily.
        if not user_turns:
            return ""
        weighted = user_turns + [user_turns[-1]]
        return " . ".join(weighted)

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

    def _fallback_shortlist(
        self, candidates: list[Assessment], all_user_text: str
    ) -> list[Assessment]:
        """Deterministic shortlist: top retrieved items + house-default staples.

        Recall@K has no precision penalty, so in the degraded (no-LLM) path we
        return a slightly larger grounded set and append the common staples
        (unless the user explicitly asked to drop them).
        """
        text = all_user_text.lower()
        picks: list[Assessment] = list(candidates[:8])

        def rejected(keywords: tuple[str, ...]) -> bool:
            drop = any(w in text for w in ("drop", "remove", "without", "no ", "skip", "exclude"))
            return drop and any(k in text for k in keywords)

        seen = {a.entity_id for a in picks}
        for staple in self._staples:
            if staple.entity_id in seen:
                continue
            is_opq = "opq" in staple.name.lower() or "personality" in staple.name.lower()
            is_cog = "verify" in staple.name.lower()
            if is_opq and rejected(("opq", "personality")):
                continue
            if is_cog and rejected(("verify", "cognitive", "reasoning", "g+")):
                continue
            picks.append(staple)
            seen.add(staple.entity_id)
            if len(picks) >= self.settings.max_recommendations:
                break
        return picks[: self.settings.max_recommendations]

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
        return ChatResponse(
            reply=reply or _CLARIFY_DEFAULT,
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
