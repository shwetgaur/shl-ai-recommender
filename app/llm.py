"""LLM provider abstraction with automatic fallback.

Providers are tried in the configured priority order until one returns text.
If none are configured/reachable, `LLMUnavailable` is raised and the agent
falls back to its deterministic policy (so the API still responds correctly).
"""
from __future__ import annotations

import json
import logging
import re

import requests

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """Raised when no configured provider could produce a response."""


def extract_json(text: str) -> dict | None:
    """Best-effort extraction of the first JSON object from an LLM response."""
    if not text:
        return None
    # Strip code fences.
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Fast path.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Scan for a balanced {...} block.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = text[start : i + 1]
                        try:
                            obj = json.loads(chunk)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
        start = text.find("{", start + 1)
    return None


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._gemini_model = None

    # ---- Provider availability -------------------------------------------
    def available_providers(self) -> list[str]:
        s = self.settings
        keys = {"gemini": s.gemini_api_key, "groq": s.groq_api_key, "openai": s.openai_api_key}
        return [p for p in s.providers if keys.get(p)]

    @property
    def has_provider(self) -> bool:
        return bool(self.available_providers())

    # ---- Public API -------------------------------------------------------
    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 20.0,
    ) -> str:
        errors: list[str] = []
        for provider in self.settings.providers:
            try:
                if provider == "gemini" and self.settings.gemini_api_key:
                    return self._gemini(system, user, temperature, max_tokens, timeout)
                if provider == "groq" and self.settings.groq_api_key:
                    return self._openai_compatible(
                        base_url="https://api.groq.com/openai/v1",
                        api_key=self.settings.groq_api_key,
                        model=self.settings.groq_model,
                        system=system, user=user,
                        temperature=temperature, max_tokens=max_tokens, timeout=timeout,
                    )
                if provider == "openai" and self.settings.openai_api_key:
                    return self._openai_compatible(
                        base_url=self.settings.openai_base_url,
                        api_key=self.settings.openai_api_key,
                        model=self.settings.openai_model,
                        system=system, user=user,
                        temperature=temperature, max_tokens=max_tokens, timeout=timeout,
                    )
            except Exception as exc:  # try the next provider
                logger.warning("Provider %s failed: %s", provider, exc)
                errors.append(f"{provider}: {exc}")
        raise LLMUnavailable("; ".join(errors) or "no LLM provider configured")

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        text = self.generate(system, user, **kwargs)
        obj = extract_json(text)
        if obj is None:
            raise LLMUnavailable("LLM returned non-JSON output")
        return obj

    # ---- Providers --------------------------------------------------------
    def _gemini(self, system, user, temperature, max_tokens, timeout) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(
            self.settings.gemini_model, system_instruction=system
        )
        resp = model.generate_content(
            user,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json",
            },
            request_options={"timeout": timeout},
        )
        return (resp.text or "").strip()

    def _openai_compatible(
        self, *, base_url, api_key, model, system, user, temperature, max_tokens, timeout
    ) -> str:
        url = base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
