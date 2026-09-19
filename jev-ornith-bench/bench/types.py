"""Core vocabulary.

The experiment hinges on one idea: a task is a sequence of *decision points*,
and the arms differ only in who answers them. Ornith always does the
open-ended generation (what command to write, what value to extract). The arm
decides the typed, discrete questions: which tool, stop or keep going, safe or
needs a human, how urgent.

Keeping that split explicit is what makes the comparison mean anything. If the
arms differed in their generation too, a win could not be attributed to
strictness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class Kind(str, Enum):
    """The three shapes a typed decision can take.

    These mirror Jev's primitives exactly (noul / choice / score) so that the
    Jev arm is a thin pass-through and the other arms are asked for precisely
    the same thing. Any richer question type would advantage the arms that can
    emit free text, which would confound the result.
    """

    NOUL = "noul"      # yes/no, answered as a probability of yes
    CHOICE = "choice"  # pick one label from a fixed set
    SCORE = "score"    # rate against an ordered rubric


@dataclass(frozen=True)
class Question:
    """One typed decision to be answered about some state."""

    name: str
    kind: Kind
    instructions: str
    # CHOICE: {label: description or None}. Jev caps this at 255 labels.
    options: dict[str, str | None] | None = None
    # SCORE: ordered rubric, one entry per level starting at 0. Jev caps at 256.
    levels: list[str] | None = None
    # NOUL: what counts as yes / no.
    true_means: str | None = None
    false_means: str | None = None

    def __post_init__(self) -> None:
        if self.kind is Kind.CHOICE:
            if not self.options:
                raise ValueError(f"choice question {self.name!r} needs options")
            if len(self.options) > 255:
                raise ValueError(f"choice question {self.name!r}: Jev allows at most 255 options")
        if self.kind is Kind.SCORE:
            if not self.levels:
                raise ValueError(f"score question {self.name!r} needs levels")
            if len(self.levels) > 256:
                raise ValueError(f"score question {self.name!r}: Jev allows at most 256 levels")

    @property
    def labels(self) -> list[str]:
        """The admissible answers, for grading and for violation detection."""
        if self.kind is Kind.CHOICE:
            assert self.options is not None
            return list(self.options)
        if self.kind is Kind.NOUL:
            return ["true", "false"]
        assert self.levels is not None
        return [str(i) for i in range(len(self.levels))]


@dataclass
class Outcome:
    """What an arm returned for one question.

    `violation` is the load-bearing field. It is True when the arm produced
    something outside the declared type -- an invented tool name, unparseable
    JSON, a score off the end of the rubric. Jev cannot set it; the solo arm
    can. Separating this from `correct` is the whole point: Jev trivially wins
    on violations, and the interesting question is whether that converts into
    task success.
    """

    question: str
    kind: Kind
    value: Any                       # str label, bool, or float score
    confidence: float | None = None  # Jev choice/score confidence; None if unavailable
    probabilities: dict[str, float] | None = None
    latency_ms: float = 0.0
    violation: bool = False          # produced something outside the type
    abstained: bool = False          # declined / escalated under a confidence gate
    error: str | None = None
    raw: str | None = None           # the arm's unparsed output, for forensics

    @property
    def label(self) -> str:
        """The answer normalised to a string, so arms compare like for like."""
        if self.kind is Kind.NOUL:
            return "true" if bool(self.value) else "false"
        if self.kind is Kind.SCORE:
            return str(int(round(float(self.value))))
        return str(self.value)


@dataclass
class Task:
    """One graded item.

    `mode="decision"` is a single typed judgement with a known answer -- cheap,
    repeatable, and where nearly all the statistical power lives.

    `mode="agentic"` is a real loop in a sandbox graded on end state -- slow and
    noisy, but the only thing that shows whether better decisions become better
    outcomes.
    """

    id: str
    suite: str
    mode: Literal["decision", "agentic"]
    state: str                          # the context handed to the decider
    question: Question | None = None    # decision mode
    expected: str | None = None         # decision mode: the correct label
    # A decision the grader treats as acceptable-but-not-ideal. Routing a
    # ticket to a plausible neighbouring queue is not the same failure as
    # deleting the wrong file.
    acceptable: list[str] = field(default_factory=list)
    # Agentic mode.
    goal: str | None = None
    setup: str | None = None            # shell run in the sandbox before the task
    check: str | None = None            # shell run after; exit 0 means success
    tools: list[str] = field(default_factory=list)
    max_turns: int = 8
    # Any task whose correct behaviour is to refuse or escalate rather than act.
    # Measured separately: a model that never acts scores well on safety and
    # uselessly on everything else.
    is_trap: bool = False
    notes: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """One task, one arm, one repeat."""

    task_id: str
    suite: str
    arm: str
    repeat: int
    success: bool
    outcomes: list[Outcome] = field(default_factory=list)
    turns: int = 0
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    unsafe: bool = False       # took a destructive action it should not have
    escalated: bool = False
    error: str | None = None
    transcript: list[dict[str, Any]] = field(default_factory=list)

    @property
    def violations(self) -> int:
        return sum(1 for o in self.outcomes if o.violation)
