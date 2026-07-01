"""Pydantic models defining the (non-negotiable) API contract.

Request:
    { "messages": [ {"role": "user"|"assistant"|"system", "content": "..."} ] }

Response:
    {
      "reply": "...",
      "recommendations": [ {"name": "...", "url": "...", "test_type": "K"} ],
      "end_of_conversation": false
    }

`recommendations` is an empty list while clarifying or refusing, and a list of
1..10 items once a shortlist is committed.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def _normalize_role(cls, v: str) -> str:
        v = (v or "").strip().lower()
        # Be liberal in what we accept; map common aliases.
        alias = {
            "human": "user",
            "ai": "assistant",
            "bot": "assistant",
            "model": "assistant",
            "developer": "system",
        }
        v = alias.get(v, v)
        if v not in {"user", "assistant", "system"}:
            # Unknown roles are treated as user input rather than rejected,
            # so a slightly-off client never breaks the conversation.
            return "user"
        return v

    @field_validator("content")
    @classmethod
    def _coerce_content(cls, v) -> str:
        if v is None:
            return ""
        return str(v)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(default_factory=list)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str = ""


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
