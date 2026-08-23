"""FastAPI service exposing GET /health and POST /chat."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .agent import Agent
from .catalog import load_catalog
from .config import get_settings
from .llm import LLMClient
from .retrieval import HybridRetriever
from .schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("shl.api")

_STATE: dict = {}


def _build_agent() -> Agent:
    settings = get_settings()
    catalog = load_catalog(settings)
    retriever = HybridRetriever(catalog, settings)
    llm = LLMClient(settings)
    logger.info(
        "Agent ready: %d assessments | retrieval=%s | llm_providers=%s",
        len(catalog), retriever.mode, llm.available_providers() or "none(fallback)",
    )
    return Agent(catalog, retriever, llm, settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _STATE["agent"] = _build_agent()
    except Exception:
        logger.exception("Failed to build agent during startup.")
        _STATE["agent"] = None
    yield
    _STATE.clear()


app = FastAPI(
    title="SHL Conversational Assessment Recommender",
    version="1.0.0",
    description="Conversational agent that recommends SHL assessments grounded in the catalog.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_agent() -> Agent:
    agent = _STATE.get("agent")
    if agent is None:
        # Lazy rebuild if startup failed for a transient reason.
        agent = _build_agent()
        _STATE["agent"] = agent
    return agent


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    agent = _STATE.get("agent")
    return {
        "service": "SHL Conversational Assessment Recommender",
        "status": "ok" if agent else "starting",
        "endpoints": {"health": "GET /health", "chat": "POST /chat"},
    }


@app.get("/api")
def api_info() -> dict:
    agent = _STATE.get("agent")
    return {
        "service": "SHL Conversational Assessment Recommender",
        "status": "ok" if agent else "starting",
        "endpoints": {"health": "GET /health", "chat": "POST /chat", "ui": "GET /"},
    }


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # Always return a schema-valid response, even on unexpected failure.
    try:
        agent = _get_agent()
        return agent.handle(request.messages)
    except Exception:
        logger.exception("Unhandled error in /chat")
        return ChatResponse(
            reply=(
                "Sorry, I hit a temporary issue. Could you restate the role or skills "
                "you're hiring for so I can suggest SHL assessments?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )


@app.exception_handler(Exception)
async def _catch_all(request, exc):  # pragma: no cover
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=200,
        content={
            "reply": "Sorry, something went wrong. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )
