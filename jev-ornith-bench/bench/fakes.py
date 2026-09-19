"""Scripted stand-ins for both backends.

These exist so the entire pipeline -- loading, deciding, grading, statistics,
report -- can be exercised with no GPU, no API key and no network. Two
reasons that is worth the code:

1. You can confirm the harness works before spending an afternoon of GPU time
   discovering that the report divides by zero.
2. The tests can assert on the machinery instead of on a stochastic model,
   which is the only way any of it is testable at all.

The fakes are deliberately *imperfect*. The fake Ornith emits malformed
answers at a configurable rate, because a fake that always returns a clean
label would never exercise the violation path -- the one piece of behaviour
the whole experiment is built to measure.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any

from bench.types import Kind, Question


def _rng(*parts: str, seed: int = 0) -> random.Random:
    """Deterministic per-item randomness, so a smoke run is reproducible."""
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return random.Random(int(digest[:16], 16) ^ seed)


@dataclass
class _Message:
    content: str
    reasoning_content: str | None = None
    tool_calls: list[Any] = field(default_factory=list)


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Usage:
    prompt_tokens: int = 100
    completion_tokens: int = 20


@dataclass
class _Completion:
    choices: list[_Choice]
    usage: _Usage = field(default_factory=_Usage)


class FakeOrnithClient:
    """Mimics the OpenAI client surface that `bench.ornith.Ornith` uses.

    `accuracy` is the chance it picks the right answer when it can infer one;
    `malformed` is the chance it answers outside the type entirely, which only
    happens when decoding is unconstrained -- mirroring the real asymmetry
    between the solo arm and the other two.
    """

    def __init__(self, accuracy: float = 0.7, malformed: float = 0.15, seed: int = 0) -> None:
        self.accuracy = accuracy
        self.malformed = malformed
        self.seed = seed
        self.calls = 0
        self.chat = self  # so `client.chat.completions.create(...)` resolves
        self.completions = self

    def create(self, **kwargs: Any) -> _Completion:
        self.calls += 1
        messages = kwargs["messages"]
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        schema = (kwargs.get("extra_body") or {}).get("guided_json")
        rng = _rng(prompt, str(self.calls), seed=self.seed)

        labels, truth = _infer(prompt)
        if schema is not None:
            # Constrained: always in-type, and the fake honours the schema the
            # way a working server would.
            answer = truth if (truth and rng.random() < self.accuracy) else (rng.choice(labels) if labels else "0")
            payload = _coerce(answer, schema)
            return _Completion([_Choice(_Message(json.dumps({"answer": payload})))])

        if labels and rng.random() < self.malformed:
            # Out-of-type: the failure mode the solo arm exists to exhibit.
            answer = rng.choice([
                "I'd need more context to say.",
                "probably " + rng.choice(labels) + " or " + rng.choice(labels),
                "other",
                "",
            ])
        elif labels:
            answer = truth if (truth and rng.random() < self.accuracy) else rng.choice(labels)
        else:
            answer = "```sh\nls -la\n```"

        return _Completion([_Choice(_Message(f"<think>weighing it up</think>{answer}"))])


def _coerce(answer: str, schema: dict[str, Any]) -> Any:
    kind = schema.get("properties", {}).get("answer", {}).get("type")
    if kind == "boolean":
        return answer.lower() in {"true", "yes", "1"}
    if kind == "integer":
        try:
            return int(float(answer))
        except ValueError:
            return 0
    return answer


def _infer(prompt: str) -> tuple[list[str], str | None]:
    """Recover the label set from the rendered prompt, and guess a truth.

    Crude on purpose. The fake only needs to be right often enough that the
    arms separate; it is not pretending to be a model.
    """
    labels: list[str] = []
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("- ") and ":" in line:
            labels.append(line[2:].split(":", 1)[0].strip())
        elif line.startswith("- ") and len(line) < 40:
            labels.append(line[2:].strip())
    if "yes or no" in prompt.lower():
        labels = ["yes", "no"]
    numeric = [l for l in labels if l.isdigit()]
    if numeric:
        labels = numeric

    lowered = prompt.lower()
    truth = None
    for hint, answer in (
        ("no space left", "disk_full"), ("permission denied", "permissions"),
        ("rm -rf /", "no"), ("du -sh", "yes"), ("systemctl status", "yes"),
        ("charged twice", "billing"), ("502", "technical"),
    ):
        if hint in lowered and (not labels or answer in labels):
            truth = answer
            break
    if truth is None and labels:
        truth = labels[0]
    return labels, truth


# ------------------------------------------------------------------ fake Jev

@dataclass
class _NoulAnswer:
    noul: float
    type: str = "noul"


@dataclass
class _ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]
    type: str = "choice"


@dataclass
class _ScoreAnswer:
    score: float
    confidence: float
    legend: dict[int, str]
    probabilities: dict[int, float]
    type: str = "score"


@dataclass
class _Response:
    answers: dict[str, Any]
    model: str = "jev-fake"
    usage: Any = None


class FakeJevClient:
    """Mimics `TypeSafeClient.system_one`. In-type by construction, like the real thing."""

    def __init__(self, accuracy: float = 0.82, seed: int = 0) -> None:
        self.accuracy = accuracy
        self.seed = seed
        self.calls = 0

    def system_one(self, state: Any, questions: dict[str, Any], **kwargs: Any) -> _Response:
        self.calls += 1
        answers: dict[str, Any] = {}
        for name, question in questions.items():
            rng = _rng(str(state), name, str(self.calls), seed=self.seed)
            kind = getattr(question, "type", None)
            right = rng.random() < self.accuracy
            # Confidence tracks correctness, which is what "calibrated" means
            # and what makes the gate sweep produce a sensible curve.
            confidence = rng.uniform(0.75, 0.99) if right else rng.uniform(0.3, 0.7)

            if kind == "noul":
                probability = confidence if right else 1 - confidence
                answers[name] = _NoulAnswer(noul=probability)
            elif kind == "score":
                levels = list(range(len(question.criteria)))
                pick = levels[0] if right else rng.choice(levels)
                probabilities = {l: (confidence if l == pick else (1 - confidence) / max(len(levels) - 1, 1)) for l in levels}
                answers[name] = _ScoreAnswer(
                    score=float(pick), confidence=confidence,
                    legend={l: str(question.criteria[l]) for l in levels},
                    probabilities=probabilities,
                )
            else:
                options = list(question.criteria)
                pick = options[0] if right else rng.choice(options)
                probabilities = {o: (confidence if o == pick else (1 - confidence) / max(len(options) - 1, 1)) for o in options}
                answers[name] = _ChoiceAnswer(choice=pick, confidence=confidence, probabilities=probabilities)
        return _Response(answers=answers)

    def close(self) -> None:
        pass


def fake_question(question: Question) -> Any:
    """Stand-in for `bench.deciders.jev.to_question` when the SDK is absent."""

    @dataclass
    class _Q:
        type: str
        instructions: str
        criteria: Any

    if question.kind is Kind.CHOICE:
        assert question.options is not None
        return _Q("choice", question.instructions, dict(question.options))
    if question.kind is Kind.SCORE:
        return _Q("score", question.instructions, list(question.levels or []))
    return _Q("noul", question.instructions, None)
