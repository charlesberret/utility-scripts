"""The contract every arm implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from bench.types import Outcome, Question


@runtime_checkable
class Decider(Protocol):
    """Answers a typed question about some state."""

    name: str

    def decide(self, state: str, question: Question) -> Outcome:
        """Answer one question. Must never raise; failures come back as an
        Outcome with `error` set and `violation=True`, so that a crashing arm
        is scored rather than silently dropped from the run."""
        ...


def gate(outcome: Outcome, threshold: float) -> Outcome:
    """Escalate instead of acting when confidence falls below `threshold`.

    Only meaningful for an arm that reports calibrated confidence. Arms that
    report none are left alone rather than being forced to guess a number --
    inventing a confidence for them would manufacture exactly the advantage
    the experiment is trying to measure.
    """
    if threshold <= 0 or outcome.confidence is None:
        return outcome
    if outcome.confidence < threshold:
        outcome.abstained = True
    return outcome
