"""Thin wrapper over the local Ornith server.

Ornith is a reasoning model: its replies open with a <think> block. vLLM and
SGLang, launched with a reasoning parser, split that into a separate
`reasoning_content` field. Runtimes launched without one (and some Ollama
builds) leave the block inline, so we strip it defensively -- otherwise the
chain of thought lands in the parsed answer and every arm looks broken for
reasons that have nothing to do with the experiment.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from bench.config import OrnithConfig

_THINK = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


@dataclass
class Reply:
    text: str
    reasoning: str | None
    tool_calls: list[Any]
    latency_ms: float
    tokens_in: int
    tokens_out: int


def strip_reasoning(text: str) -> tuple[str, str | None]:
    """Return (answer, reasoning) for a reply with an inline <think> block."""
    if "<think>" not in text:
        return text.strip(), None
    reasoning = "\n".join(m.strip() for m in _THINK.findall(text))
    stripped = _THINK.sub("", text)
    # An unterminated block means the model ran out of tokens mid-thought.
    # Everything after the opening tag is reasoning, and there is no answer.
    if "<think>" in stripped:
        return "", stripped.split("<think>", 1)[1].strip()
    return stripped.strip(), reasoning.strip() or None


class Ornith:
    """Chat wrapper. Also used for the `solo` and `constrained` arms."""

    def __init__(self, config: OrnithConfig | None = None, client: Any = None) -> None:
        self.config = config or OrnithConfig()
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Reply:
        """One turn.

        `json_schema` drives the `constrained` control arm. We ask for it two
        ways because servers disagree: vLLM and SGLang accept `guided_json` as
        an extra body field, while llama.cpp and newer OpenAI-compatible
        shims want `response_format: json_schema`. Sending both is harmless --
        each server ignores the key it does not know -- and it means the
        control arm actually gets constrained decoding rather than silently
        degrading into the solo arm, which would quietly destroy the
        comparison.
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        extra_body: dict[str, Any] = {}
        if self.config.top_k > 0:
            extra_body["top_k"] = self.config.top_k
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_schema is not None:
            extra_body["guided_json"] = json_schema
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "decision", "schema": json_schema, "strict": True},
            }
        if extra_body:
            kwargs["extra_body"] = extra_body

        start = time.perf_counter()
        response = self._client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - start) * 1000

        message = response.choices[0].message
        text = message.content or ""
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None:
            text, reasoning = strip_reasoning(text)
        usage = getattr(response, "usage", None)
        return Reply(
            text=text,
            reasoning=reasoning,
            tool_calls=list(getattr(message, "tool_calls", None) or []),
            latency_ms=latency,
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
        )
