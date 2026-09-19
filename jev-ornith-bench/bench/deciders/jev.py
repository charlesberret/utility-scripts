"""Arm 3: Jev, TypeSafe's System One model.

Jev does not generate text. It answers typed questions about a state and
returns calibrated probabilities, so an out-of-type answer is not something it
declines to produce -- it is not in the output space at all. Our `Question`
type mirrors Jev's three primitives one-for-one, which keeps this arm a thin
translation rather than a reimplementation.

We call `typesafe_sdk` directly rather than the higher-level `jev` decorator
package. The decorator is nicer to write, but it thresholds bools, takes the
argmax of choices and discards the probabilities -- and its own documentation
says to drop to the SDK when you need confidence, which is precisely the
signal this experiment is built to evaluate.
"""

from __future__ import annotations

import time
from typing import Any

from bench.config import JevConfig
from bench.types import Kind, Outcome, Question


def to_question(question: Question) -> Any:
    """Translate our Question into a TypeSafe question object."""
    from typesafe_sdk import Choice, Noul, Score

    if question.kind is Kind.CHOICE:
        assert question.options is not None
        return Choice(instructions=question.instructions, criteria=dict(question.options))
    if question.kind is Kind.NOUL:
        criteria = None
        if question.true_means or question.false_means:
            criteria = {"true": question.true_means, "false": question.false_means}
        return Noul(instructions=question.instructions, criteria=criteria)
    assert question.levels is not None
    return Score(instructions=question.instructions, criteria=list(question.levels))


class JevDecider:
    name = "jev"

    def __init__(
        self,
        config: JevConfig | None = None,
        client: Any = None,
        *,
        bool_threshold: float = 0.5,
        question_builder: Any = None,
    ) -> None:
        self.config = config or JevConfig()
        self._client = client
        self._owned = client is None
        self.bool_threshold = bool_threshold
        # Injectable so the offline smoke run can build questions without
        # importing the SDK at all.
        self._build = question_builder or to_question

    def _get(self) -> Any:
        if self._client is None:
            from typesafe_sdk import TypeSafeClient

            kwargs: dict[str, Any] = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = TypeSafeClient(**kwargs)
        return self._client

    def decide(self, state: str, question: Question) -> Outcome:
        start = time.perf_counter()
        try:
            response = self._get().system_one(
                state=state,
                questions={question.name: self._build(question)},
                model=self.config.model,
                timeout=self.config.timeout,
            )
        except Exception as exc:
            return Outcome(
                question=question.name, kind=question.kind, value=None,
                violation=True, error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        latency = (time.perf_counter() - start) * 1000
        out = Outcome(question=question.name, kind=question.kind, value=None, latency_ms=latency)
        answer = response.answers.get(question.name)
        if answer is None:
            out.violation = True
            out.error = "no answer returned for question"
            return out

        if question.kind is Kind.NOUL:
            # `noul` is p(yes). Threshold it into the bool the graders compare
            # against, but keep the raw probability in `probabilities` -- that
            # is what the calibration report scores, and rounding it away here
            # would throw out Jev's most distinctive output.
            probability = float(answer.noul)
            out.value = probability >= self.bool_threshold
            out.probabilities = {"true": probability, "false": 1.0 - probability}
            # Distance from the fence, mapped to 0..1, so a noul can be gated
            # on the same scale as a choice. Jev reports no confidence field
            # for nouls, and this is a derived stand-in rather than a number
            # the API gave us -- it is documented as such in the README.
            out.confidence = abs(probability - 0.5) * 2
        elif question.kind is Kind.SCORE:
            out.value = float(answer.score)
            out.confidence = float(answer.confidence)
            out.probabilities = {str(k): float(v) for k, v in answer.probabilities.items()}
        else:
            out.value = str(answer.choice)
            out.confidence = float(answer.confidence)
            out.probabilities = {str(k): float(v) for k, v in answer.probabilities.items()}

        usage = getattr(response, "usage", None)
        out.raw = f"model={getattr(response, 'model', '?')} in={getattr(usage, 'input_tokens', None)} out={getattr(usage, 'output_tokens', None)}"
        return out

    def close(self) -> None:
        if self._owned and self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
