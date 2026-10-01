"""Self-hosted LLM client (OpenAI-compatible chat API).

This works with Ollama (``/v1``), vLLM, llama.cpp ``server``, LocalAI and TGI, serving open
weights such as Qwen2.5-7B/14B-Instruct or Llama-3.1-8B-Instruct. No data leaves the
operator's infrastructure.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import httpx

from himsat.config import Settings, get_settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMClient:
    base_url: str
    model: str
    api_key: str = "not-needed"
    timeout_s: float = 90.0
    temperature: float = 0.2
    _mode: str | None = None  # structured-output mode the server accepted last

    @classmethod
    def from_settings(cls, s: Settings | None = None) -> LLMClient | None:
        s = s or get_settings()
        if s.llm_backend == "none":
            return None
        return cls(s.llm_base_url, s.llm_model, s.llm_api_key, s.llm_timeout_s, s.llm_temperature)

    @property
    def name(self) -> str:
        return f"llm:{self.model}"

    def chat_json(self, system: str, user: str, max_tokens: int = 1800, schema: dict | None = None) -> dict:
        """Chat completion that must return a JSON object.

        Servers differ in structured-output support (OpenAI / vLLM / LM Studio: ``json_schema``;
        Ollama / llama.cpp: ``json_object``; some: neither). Modes are tried in that order and
        the first one the server accepts is remembered.
        """
        if "qwen3" in self.model.lower():
            user = user + " /no_think"  # Qwen3 soft switch: no hidden reasoning, faster and shorter
        base = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        modes = ["json_schema", "json_object", "none"] if schema else ["json_object", "none"]
        if self._mode in modes:
            modes = [self._mode] + [m for m in modes if m != self._mode]
        url = self.base_url.rstrip("/") + "/chat/completions"
        last: Exception | None = None
        for mode in modes:
            payload = dict(base)
            if mode == "json_schema":
                payload["response_format"] = {"type": "json_schema",
                                              "json_schema": {"name": "alert", "strict": True, "schema": schema}}
            elif mode == "json_object":
                payload["response_format"] = {"type": "json_object"}
            try:
                r = httpx.post(url, json=payload, timeout=self.timeout_s,
                               headers={"Authorization": f"Bearer {self.api_key}"})
            except httpx.HTTPError as e:
                raise LLMError(f"LLM request failed: {e}") from e
            if r.status_code in (400, 422) and mode != "none" and (
                    "response_format" in r.text or "json" in r.text.lower()):
                last = LLMError(f"{mode} unsupported: {r.text[:200]}")
                continue
            try:
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
                raise LLMError(f"LLM request failed: {e}; body={r.text[:300]}") from e
            self._mode = mode
            return parse_json_object(content)
        raise LLMError(f"LLM server rejected all response formats: {last}")

    def healthy(self) -> bool:
        try:
            r = httpx.get(self.base_url.rstrip("/") + "/models", timeout=5,
                          headers={"Authorization": f"Bearer {self.api_key}"})
            return r.status_code == 200
        except httpx.HTTPError:
            return False


def parse_json_object(text: str) -> dict:
    text = text.strip()
    # strip reasoning blocks and markdown fences some models emit
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise LLMError("LLM output contains no JSON object") from None
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM output is not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise LLMError("LLM output JSON is not an object")
    return obj
