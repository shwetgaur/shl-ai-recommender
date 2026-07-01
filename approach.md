# Approach: Conversational SHL Assessment Recommender

## Problem framing
The agent must move a user from vague intent to a grounded shortlist of SHL
assessments over a short (≤8-turn), non-deterministic, **stateless** conversation.
The four required behaviors — clarify, recommend, refine, compare — plus staying
in scope, all reduce to one core loop each turn: *screen for safety → retrieve
grounded candidates → decide an action → return a schema-valid response*. I
optimized for the scoring surface: hard evals (schema, catalog-only, turn cap),
Recall@10 on final shortlists, and behavior probes.

## Architecture and stack
FastAPI + Pydantic for a strict, self-documenting contract. The request handler
is a thin wrapper around a stateless `Agent`. Three deliberate layers give
robustness:

1. **Deterministic safety screen** (`safety.py`) — high-precision regex for
   prompt-injection, off-topic, and legal/compliance requests. This guarantees
   the refusal behavior probes pass *even with no LLM available*.
2. **LLM policy** (`agent.py` + `prompts.py`) — a single JSON-constrained LLM
   call per turn that chooses `clarify | recommend | compare | refuse` and
   selects assessments **by catalog ID from a retrieved candidate set**.
3. **Deterministic fallback** — if no key is set or the LLM errors/times out, a
   retrieval-driven policy still returns a grounded, schema-valid shortlist.

**Statelessness / refinement.** I re-derive the full shortlist from the entire
conversation every turn rather than storing state. Because the whole history
(including "add personality", "drop REST, add AWS") is replayed to retrieval +
LLM, edits update the list instead of restarting — which is exactly what the
"honors edits" probe checks, and it needs zero server-side state.

**Grounding by construction.** The LLM never emits URLs. It returns catalog IDs
that I resolve against the normalized catalog; invalid IDs are dropped and, if
needed, backfilled from retrieval. Hallucinated URLs are therefore structurally
impossible — the strongest hard-eval guarantee.

## Catalog & retrieval
The 377-item catalog is normalized once into typed records; `test_type` letters
(A/B/C/D/E/K/P/S) are mapped from the human-readable `keys` field (verified
against the sample traces). Lookups are indexed by ID, URL (trailing-slash
tolerant), and name. The source JSON contains raw control characters, so parsing
uses `strict=False` — a real edge case that would crash a naive loader.

Retrieval is **hybrid**: BM25 (lexical, excellent for exact skill terms like
`docker`, `c#`, `opq`) blended with MiniLM sentence-embeddings (semantic, needed
for "senior leadership" → OPQ). Catalog embeddings are precomputed and cached to
disk (fingerprinted by model + catalog), so only the short query is embedded per
request (~tens of ms). If `sentence-transformers` can't load (low-RAM tiers), the
retriever transparently degrades to BM25-only — the service never fails on
retrieval. A custom tokenizer preserves tech tokens (`c++`, `.net`).

## Prompt design
The system prompt encodes the hard rules (scope, catalog-only, 1–10 items, no
recommend on a vague opener, re-derive on edits) and **house-composition
guidance** distilled from the traces: good batteries layer knowledge + cognitive
+ personality, and OPQ32r / Verify G+ are common defaults unless the user opts
out. Each turn the user prompt carries the conversation, a compact candidate
block (ID, name, type, keys, duration, levels, languages, description, URL), and
a turn hint (vague-opener vs must-commit-now). Output is a fixed JSON object,
parsed with a resilient brace-matching extractor.

## Evaluation
`scripts/evaluate.py` parses the 10 sample traces, extracts the gold final
shortlist from each trace's last table, replays the user turns through the agent,
and computes Recall@10, plus seven behavior probes. Measured results:

- **Behavior probes: 7/7 pass** in every mode (incl. no-LLM).
- **Mean Recall@10 on the 10 public traces: 0.69** with the LLM policy (Groq
  `llama-3.3-70b`, hybrid retrieval); **0.61** in the no-LLM deterministic floor.
- 32 unit/integration tests (schema, tolerant parsing, safety, grounding,
  turn-cap commit, LLM-path via a scripted fake LLM, LLM-failure fallback).

**What didn't work / iterations (measured).** (1) The raw LLM policy actually
*underperformed* the deterministic floor (0.42 vs 0.61): it curated tight 3–5
item lists and, being stateless, sometimes returned nothing on a confirmation
turn. Fixes, each measured: embed the shortlist in every reply so history carries
it forward, carry it forward on confirmation, and add house-default staples →
0.57. (2) Capping the shortlist at 8 left recall on the table; since Recall@K has
no precision penalty, backfilling to the full 10 with grounded, on-topic,
non-removed items → 0.67. (3) A coarse "any removal disables backfill" flag
wrongly suppressed Verify G+ in C9 (user said "drop REST", not "drop Verify"); a
per-item removed-terms filter honors edits precisely → 0.69 and keeps the
edit-probe green. (4) Doubling the last user turn in the retrieval query added
noise when that turn was a confirmation; dropping confirmation-only turns from
the query helped refinement traces. (5) Pure lexical retrieval missed semantic
queries (leadership→OPQ); the embedding blend fixed C1/C3. (6) An over-eager
legal regex flagged benign "compliance" queries — tightened to require an
obligation verb, with a test asserting legitimate queries are never flagged.

## Reliability / edge cases
`/chat` always returns a valid `ChatResponse` (200) — agent, request handler,
and a catch-all handler each degrade gracefully; malformed roles are coerced,
empty/missing messages yield a clarify. Per-request LLM timeout is bounded well
under the 30s evaluator cap, and the turn-cap rule forces a commit on the last
allowed turn.

## AI tools used
Used an AI coding assistant (Cursor) to scaffold boilerplate, draft the test
matrix, and speed up iteration. All design decisions — the three-layer
robustness model, grounding-by-ID, stateless re-derivation, hybrid retrieval,
and the evaluation harness — are my own and are reflected directly in the code.
