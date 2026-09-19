"""Arm 1: Ornith alone, prompted in prose and parsed.

This is how most people actually wire a small local model into a decision, and
it is the only arm that can produce something outside the declared type. Its
violation rate is the quantity the other two arms drive to zero by
construction.

The parser here is deliberately generous: it accepts bare labels, quoted
labels, trailing full stops, `answer: x` preambles, and a unique substring
match. Every bit of that leniency flatters the baseline. That is the right
direction to err -- if Jev still wins against a forgiving parser, the win is
about judgement rather than about our own string handling.
"""

from __future__ import annotations

import json
import re

from bench.config import OrnithConfig
from bench.ornith import Ornith
from bench.types import Kind, Outcome, Question

_PREAMBLE = re.compile(r"^\s*(?:the\s+)?(?:answer|label|choice|category|score|result)\s*(?:is)?\s*[:\-]?\s*", re.I)


def _clean(text: str) -> str:
    text = text.strip()
    # Models love to wrap a one-word answer in a fenced block.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    # A JSON object slipped out despite a prose prompt: take the obvious field.
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            for key in ("answer", "label", "choice", "category", "value", "score", "result"):
                if key in obj:
                    return str(obj[key]).strip()
    text = _PREAMBLE.sub("", text)
    return text.strip().strip("\"'`*.,;: \n\t")


def _match(text: str, labels: list[str]) -> str | None:
    """Resolve free text to one admissible label, or None if it is outside the type."""
    cleaned = _clean(text)
    lowered = cleaned.lower()
    for label in labels:
        if lowered == label.lower():
            return label
    # First line only -- a model that answers then explains should still count.
    first = lowered.split("\n", 1)[0].strip().strip("\"'`*.,;: ")
    for label in labels:
        if first == label.lower():
            return label
    # Unique substring. Ambiguous mentions ("billing or technical") stay a
    # violation, which is the honest reading: no single answer was given.
    hits = [label for label in labels if re.search(rf"\b{re.escape(label.lower())}\b", lowered)]
    if len(hits) == 1:
        return hits[0]
    return None


_YES = {"yes", "true", "y", "1", "affirmative", "correct"}
_NO = {"no", "false", "n", "0", "negative", "incorrect"}


def _prompt(question: Question) -> str:
    if question.kind is Kind.CHOICE:
        assert question.options is not None
        lines = [
            f"- {label}" + (f": {desc}" if desc else "")
            for label, desc in question.options.items()
        ]
        return (
            f"{question.instructions}\n\nChoose exactly one:\n" + "\n".join(lines) +
            "\n\nReply with the label alone and nothing else."
        )
    if question.kind is Kind.NOUL:
        extra = ""
        if question.true_means:
            extra += f"\nyes means: {question.true_means}"
        if question.false_means:
            extra += f"\nno means: {question.false_means}"
        return f"{question.instructions}{extra}\n\nReply with exactly one word: yes or no."
    assert question.levels is not None
    rubric = "\n".join(f"- {i}: {text}" for i, text in enumerate(question.levels))
    return (
        f"{question.instructions}\n\nRate using this scale:\n{rubric}\n\n"
        f"Reply with the number alone (0 to {len(question.levels) - 1}) and nothing else."
    )


class SoloDecider:
    name = "solo"

    def __init__(self, config: OrnithConfig | None = None, ornith: Ornith | None = None) -> None:
        self.ornith = ornith or Ornith(config)

    def decide(self, state: str, question: Question) -> Outcome:
        messages = [
            {
                "role": "system",
                "content": "You answer classification questions. Reply with the answer alone, no explanation.",
            },
            {"role": "user", "content": f"{state}\n\n---\n\n{_prompt(question)}"},
        ]
        try:
            reply = self.ornith.chat(messages, max_tokens=1024)
        except Exception as exc:  # a dead endpoint is a result, not a crash
            return Outcome(
                question=question.name, kind=question.kind, value=None,
                violation=True, error=f"{type(exc).__name__}: {exc}",
            )

        out = Outcome(
            question=question.name, kind=question.kind, value=None,
            latency_ms=reply.latency_ms, raw=reply.text,
        )
        cleaned = _clean(reply.text)

        if question.kind is Kind.NOUL:
            lowered = cleaned.lower().split("\n", 1)[0].strip().strip("\"'`*.,;: ")
            if lowered in _YES:
                out.value = True
            elif lowered in _NO:
                out.value = False
            else:
                out.violation = True
            return out

        if question.kind is Kind.SCORE:
            assert question.levels is not None
            found = re.search(r"-?\d+(?:\.\d+)?", cleaned)
            if not found:
                out.violation = True
                return out
            value = float(found.group())
            # Off the end of the rubric is exactly the failure this arm is
            # here to exhibit; do not clamp it away.
            if not 0 <= value <= len(question.levels) - 1:
                out.violation = True
                return out
            out.value = value
            return out

        label = _match(reply.text, question.labels)
        if label is None:
            out.violation = True
            return out
        out.value = label
        return out
