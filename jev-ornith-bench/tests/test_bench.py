"""Tests for the parts that would corrupt a result silently if they broke.

Nothing here touches a network or a GPU. The priority is the grading and
statistics: a parser bug shows up as a weird number you might investigate,
whereas a statistics bug shows up as a confident conclusion you would not.
"""

from __future__ import annotations

import math

import pytest

from bench.deciders.constrained import schema_for
from bench.deciders.solo import _match, SoloDecider
from bench.fakes import FakeJevClient, FakeOrnithClient, fake_question
from bench.ornith import Ornith, strip_reasoning
from bench.runner import extract_command, run_decision
from bench.stats import brier, cluster_bootstrap, ece, mcnemar, min_detectable_effect
from bench.suites import load_file
from bench.types import Kind, Outcome, Question, Task


# ------------------------------------------------------------------ parsing

@pytest.mark.parametrize("text,want", [
    ("billing", "billing"),
    ("  Billing.  ", "billing"),
    ("The answer is: technical", "technical"),
    ("```\nsales\n```", "sales"),
    ('{"answer": "billing"}', "billing"),
    ("technical - the dashboard is erroring", "technical"),
])
def test_match_accepts_reasonable_shapes(text, want):
    assert _match(text, ["billing", "technical", "sales"]) == want


@pytest.mark.parametrize("text", ["billing or technical", "plumbing", "", "I'm not sure"])
def test_match_rejects_out_of_type(text):
    """Ambiguity is a violation, not a coin flip. Silently picking one of two
    mentioned labels would hide the solo arm's real failure rate."""
    assert _match(text, ["billing", "technical", "sales"]) is None


def test_strip_reasoning_handles_unterminated_block():
    answer, reasoning = strip_reasoning("<think>ran out of tokens")
    assert answer == ""
    assert "ran out" in reasoning


def test_extract_command_skips_comments():
    assert extract_command("```sh\n# check it\ndf -h\n```") == "df -h"


def test_extract_command_returns_none_without_one():
    assert extract_command("DONE, nothing to do here") is None


# ------------------------------------------------------------------- schema

def test_schema_pins_choice_to_enum():
    question = Question("q", Kind.CHOICE, "pick", options={"a": None, "b": None})
    assert schema_for(question)["properties"]["answer"]["enum"] == ["a", "b"]


def test_schema_bounds_score_to_rubric():
    question = Question("q", Kind.SCORE, "rate", levels=["lo", "mid", "hi"])
    answer = schema_for(question)["properties"]["answer"]
    assert (answer["minimum"], answer["maximum"]) == (0, 2)


def test_question_rejects_oversized_choice():
    """Jev caps a choice at 255 options; failing at construction beats failing
    mid-sweep with a vendor error."""
    with pytest.raises(ValueError):
        Question("q", Kind.CHOICE, "pick", options={str(i): None for i in range(256)})


# ------------------------------------------------------------------ grading

def _task(**kwargs):
    base = dict(
        id="t1", suite="s", mode="decision", state="x",
        question=Question("q", Kind.CHOICE, "pick", options={"a": None, "b": None}),
        expected="a",
    )
    base.update(kwargs)
    return Task(**base)


class _Fixed:
    """A decider that returns exactly what a test needs."""

    name = "fixed"

    def __init__(self, outcome):
        self._outcome = outcome

    def decide(self, state, question):
        return self._outcome


def test_acceptable_answer_counts_as_success():
    task = _task(acceptable=["b"])
    result = run_decision(task, _Fixed(Outcome("q", Kind.CHOICE, "b")), 0)
    assert result.success


def test_violation_is_a_failure():
    result = run_decision(_task(), _Fixed(Outcome("q", Kind.CHOICE, None, violation=True)), 0)
    assert not result.success


def test_escalating_is_right_on_a_trap_and_wrong_otherwise():
    """Otherwise 'always escalate' would be the winning strategy, which would
    make the safety numbers meaningless."""
    abstained = Outcome("q", Kind.CHOICE, "a", abstained=True)
    assert run_decision(_task(is_trap=True), _Fixed(abstained), 0).success
    assert not run_decision(_task(is_trap=False), _Fixed(abstained), 0).success


def test_noul_probability_is_thresholded_not_truthy():
    """A Jev noul comes back as p(yes). Coercing 0.3 with bool() would read as
    'true' -- the single most damaging bug this harness could have."""
    from bench.deciders.jev import JevDecider

    decider = JevDecider(client=FakeJevClient(accuracy=1.0), question_builder=fake_question)
    question = Question("q", Kind.NOUL, "is it so?")
    outcome = decider.decide("state", question)
    assert isinstance(outcome.value, bool)
    assert outcome.label in {"true", "false"}
    assert outcome.probabilities is not None
    assert outcome.label == ("true" if outcome.probabilities["true"] >= 0.5 else "false")


# --------------------------------------------------------------- statistics

def test_bootstrap_resamples_tasks_not_repeats():
    """Ten repeats of one task must not look like a hundred observations. If
    this regresses, every interval in the report becomes falsely tight."""
    one_task = cluster_bootstrap({"t": [True] * 5 + [False] * 5}, resamples=500)
    assert math.isnan(one_task.lo)  # a single cluster carries no information

    # Tasks must differ from one another, or there is nothing for a
    # task-level resample to vary over and the interval collapses to a point
    # -- which is itself the correct behaviour, and the reason this fixture
    # is heterogeneous.
    mixed = {f"t{i}": [i % 3 == 0] * 2 for i in range(30)}
    wide = cluster_bootstrap(mixed, resamples=500)
    assert wide.hi - wide.lo > 0
    assert wide.lo <= wide.point <= wide.hi

    # And the interval must narrow as tasks are added, not as repeats are.
    few = cluster_bootstrap({k: mixed[k] for k in list(mixed)[:6]}, resamples=500)
    many = cluster_bootstrap({f"t{i}": [i % 3 == 0] * 2 for i in range(120)}, resamples=500)
    assert (many.hi - many.lo) < (few.hi - few.lo)


def test_mcnemar_is_symmetric_when_arms_tie():
    a = {"x": True, "y": False}
    b = {"x": False, "y": True}
    assert mcnemar(a, b).p == 1.0


def test_mcnemar_detects_a_clean_sweep():
    a = {f"t{i}": True for i in range(10)}
    b = {f"t{i}": False for i in range(10)}
    assert mcnemar(a, b).p < 0.01


def test_brier_rewards_confident_correctness():
    confident_right = brier([({"a": 1.0, "b": 0.0}, "a")])
    hedged = brier([({"a": 0.5, "b": 0.5}, "a")])
    confident_wrong = brier([({"a": 0.0, "b": 1.0}, "a")])
    assert confident_right < hedged < confident_wrong


def test_ece_flags_overconfidence():
    assert ece([(0.99, False)] * 10) > 0.9
    assert ece([(0.5, True)] * 5 + [(0.5, False)] * 5) < 0.05


def test_mde_shrinks_with_tasks_only():
    assert min_detectable_effect(200) < min_detectable_effect(50) < min_detectable_effect(12)


# ------------------------------------------------------------------- suites

def test_shipped_suites_all_load():
    from bench.suites import load

    tasks = load(["suites"])
    assert len(tasks) > 40
    assert {t.mode for t in tasks} == {"decision", "agentic"}


def test_loader_rejects_an_expected_answer_outside_the_options(tmp_path):
    """A typo'd label looks exactly like every arm failing that item."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        "suite: bad\ntasks:\n  - id: x\n    state: s\n    expected: nonexistent\n"
        "    question:\n      kind: choice\n      instructions: pick\n      options: [a, b]\n"
    )
    with pytest.raises(ValueError, match="not among"):
        load_file(path)


# -------------------------------------------------------------- integration

def test_solo_arm_produces_violations_and_constrained_does_not():
    """The core asymmetry the experiment rests on."""
    ornith = Ornith(client=FakeOrnithClient(malformed=0.9, seed=1))
    solo = SoloDecider(ornith=ornith)
    question = Question("q", Kind.CHOICE, "Which team?",
                        options={"billing": None, "technical": None})
    outcomes = [solo.decide(f"ticket {i}", question) for i in range(30)]
    assert any(o.violation for o in outcomes)

    from bench.deciders.constrained import ConstrainedDecider

    constrained = ConstrainedDecider(
        ornith=Ornith(client=FakeOrnithClient(malformed=0.9, seed=1)), two_step=False
    )
    assert not any(constrained.decide(f"ticket {i}", question).violation for i in range(30))


def test_report_builds_from_a_full_fake_run():
    from bench.cli import _deciders, _execute
    from bench.config import RunConfig
    from bench.report import build
    from bench.suites import load

    tasks = [t for t in load(["suites"]) if t.mode == "decision"][:8]
    run = RunConfig(repeats=2)
    results = _execute(tasks, _deciders(["solo", "constrained", "jev"], run, fake=True), run, fake=True)
    report = build(results, tasks, {"run_id": "test"})
    assert "Where the difference comes from" in report
    assert "Paired comparisons" in report
    assert len(results) == 8 * 3 * 2
