---
title: SHL Conversational Assessment Recommender
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SHL Conversational Assessment Recommender

A conversational agent that takes a hiring manager from a vague intent
("I'm hiring a Java developer") to a **grounded shortlist of SHL assessments**
through dialogue. It clarifies when the request is vague, recommends 1–10
assessments once it has enough context, refines when constraints change,
compares assessments on request, and stays strictly in scope (SHL catalog only).

Built for the SHL Labs AI Intern take-home assignment.

---

## Quickstart

```bash
# 1. Install
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure (optional but recommended - enables the LLM policy)
cp .env.example .env
#   set GEMINI_API_KEY (free: https://aistudio.google.com/app/apikey)
#   or  GROQ_API_KEY   (free: https://console.groq.com/keys)

# 3. Run
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 4. Try it
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java developer who works with stakeholders"}]}'
```

> The service **runs without any API key** using a deterministic, retrieval-driven
> fallback (schema-valid replies, grounded shortlists). Adding an LLM key unlocks
> the full clarify/compare/refine conversational behavior.

---

## API

### `GET /health`
Returns `{"status": "ok"}` with HTTP 200.

### `POST /chat`
Stateless — every call carries the full conversation history.

Request:
```json
{ "messages": [ {"role": "user", "content": "Hiring a Java developer"} ] }
```

Response:
```json
{
  "reply": "Sure - what seniority level are you hiring for?",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is an **empty list** while clarifying or refusing, and a list
  of **1–10 items** once a shortlist is committed. Every `url` comes from the
  scraped catalog.
- `test_type` is SHL's single-letter code (A, B, C, D, E, K, P, S), derived from
  the catalog's `keys`.
- `end_of_conversation` is `true` only when the task is complete.

---

## How it works

```
POST /chat
   │
   ├─ 1. Safety screen (deterministic)  ── injection / off-topic / legal → refuse
   │
   ├─ 2. Hybrid retrieval               ── BM25 ⊕ MiniLM embeddings over the catalog
   │
   ├─ 3. LLM policy (grounded)          ── clarify | recommend | compare | refuse
   │        selects only from retrieved candidate IDs
   │
   └─ 4. Deterministic fallback         ── used if no LLM / LLM error; still grounded
```

- **Stateless re-derivation.** Each turn the full shortlist is re-derived from the
  entire conversation, so "add personality tests" or "drop REST, add AWS" updates
  the list rather than starting over.
- **Grounding.** The LLM only ever returns catalog IDs; names/URLs/test types are
  resolved by us from the catalog, so hallucinated URLs are structurally impossible.
- **Turn cap aware.** The evaluator caps conversations at 8 messages; on the final
  turn the agent commits a shortlist instead of asking again.

See [`approach.md`](approach.md) for the full design write-up.

---

## Project layout

```
app/
  main.py        FastAPI app (/health, /chat)
  schemas.py     Strict request/response models (the API contract)
  config.py      Env-driven settings with safe defaults
  catalog.py     Catalog load/normalize/index + test_type mapping
  retrieval.py   Hybrid BM25 + embedding retriever (graceful fallback)
  llm.py         Multi-provider LLM client (Gemini/Groq/OpenAI) + JSON parsing
  safety.py      Injection / off-topic / legal guardrails
  agent.py       Stateless orchestration (clarify/recommend/refine/compare/refuse)
  prompts.py     System + user prompt templates
scripts/
  build_catalog.py   Refresh catalog + warm embedding cache (run before deploy)
  evaluate.py        Recall@10 + behavior-probe harness over the sample traces
tests/               Pytest suite (catalog, safety, agent, API, LLM path)
data/                Cached catalog + sample conversations
Dockerfile           Container for HF Spaces / Render / Fly / Railway
```

---

## Evaluation

```bash
python -m scripts.evaluate        # Recall@10 on the 10 sample traces + behavior probes
pytest                            # unit + integration tests
```

The harness parses each sample trace, replays the user turns against the agent,
and compares the final shortlist URLs to the trace's gold set (Recall@10). It also
runs behavior probes (refuse off-topic/legal/injection, no-recommend-on-vague-turn-1,
schema validity, catalog grounding).

---

## Deployment (free tiers)

**Hugging Face Spaces (recommended — 16 GB RAM, supports embeddings):**
Create a Docker Space, push this repo, set `GEMINI_API_KEY` (or `GROQ_API_KEY`) as a
Space secret. The container listens on `7860`.

**Render / Fly / Railway:** Use the included `Dockerfile`. On memory-constrained
free tiers set `RETRIEVAL_MODE=lexical` (BM25-only) to stay within RAM limits; see
`render.yaml`.

`GET /health` is the readiness probe (first cold-start call may take up to ~2 min).

---

## Configuration

All settings are environment variables (see `.env.example`). Highlights:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` / `GROQ_API_KEY` / `OPENAI_API_KEY` | – | LLM providers (tried in `LLM_PROVIDER_PRIORITY` order) |
| `RETRIEVAL_MODE` | `auto` | `auto` \| `hybrid` \| `lexical` |
| `HYBRID_SEMANTIC_WEIGHT` | `0.55` | Blend of semantic vs lexical score |
| `RETRIEVAL_TOP_K` | `40` | Candidates surfaced to the LLM per turn |
| `MAX_TURNS` | `8` | Evaluator turn cap the agent respects |
| `MAX_RECOMMENDATIONS` | `10` | Upper bound on shortlist size |
