"""Turning results into something you can actually conclude from.

The report is opinionated about one thing above all: it never prints a bare
accuracy comparison between `solo` and `jev`. That number is the one everybody
wants and it is the one that means the least, because it bundles together two
completely different effects --

  * Jev cannot emit an invalid answer, so every malformed reply the solo arm
    produced becomes a win for free; and
  * Jev might simply make better judgements.

Only the second is interesting, and only the second survives if you ever fix
your parser. So the tables below always decompose the gap, and the ladder
solo -> constrained -> jev is printed as two separate steps.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from bench.stats import brier, cluster_bootstrap, ece, mcnemar, min_detectable_effect
from bench.types import Kind, Task, TaskResult


def _by_task(results: list[TaskResult], arm: str, predicate: Any = None) -> dict[str, list[bool]]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        if r.arm != arm:
            continue
        grouped[r.task_id].append(predicate(r) if predicate else r.success)
    return dict(grouped)


def _majority(results: list[TaskResult], arm: str) -> dict[str, bool]:
    """Collapse repeats to one verdict per task, for the paired tests."""
    grouped = _by_task(results, arm)
    return {tid: sum(v) * 2 > len(v) for tid, v in grouped.items()}


def build(results: list[TaskResult], tasks: list[Task], meta: dict[str, Any]) -> str:
    by_id = {t.id: t for t in tasks}
    arms = sorted({r.arm for r in results}, key=lambda a: ["solo", "constrained", "jev"].index(a) if a in ("solo", "constrained", "jev") else 99)
    suites = sorted({r.suite for r in results})
    out: list[str] = []
    w = out.append

    w("# Jev x Ornith: does strictness make a local model more capable?")
    w("")
    w(f"Run {meta.get('run_id', '?')} at {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    w("")
    w("| | |")
    w("|---|---|")
    for key in ("ornith_model", "ornith_base_url", "jev_model", "temperature", "repeats", "confidence_gate"):
        if key in meta:
            w(f"| {key} | `{meta[key]}` |")
    w("")

    # --- power warning, before any number that invites over-reading ---------
    n_tasks = len({r.task_id for r in results})
    mde = min_detectable_effect(n_tasks)
    w(f"> **{n_tasks} tasks.** A suite this size can reliably resolve a difference of "
      f"about **{mde:.0%}** or larger. Smaller gaps than that are consistent with noise "
      f"however many repeats you run -- repeats sharpen each task's estimate but the "
      f"unit of independent information is the task. If the gap you care about is "
      f"below this line, the answer is more tasks, not more passes.")
    w("")

    # --- headline ------------------------------------------------------------
    w("## Task success")
    w("")
    w("Intervals are 95% percentile bootstrap, resampling *tasks* (not repeats), because")
    w("repeats within a task are correlated and treating them as independent would")
    w("shrink these intervals by roughly the square root of the repeat count.")
    w("")
    w("| suite | " + " | ".join(arms) + " |")
    w("|---|" + "---|" * len(arms))
    for suite in suites:
        scoped = [r for r in results if r.suite == suite]
        cells = [str(cluster_bootstrap(_by_task(scoped, arm))) for arm in arms]
        w(f"| {suite} | " + " | ".join(cells) + " |")
    overall = [str(cluster_bootstrap(_by_task(results, arm))) for arm in arms]
    w("| **all** | " + " | ".join(f"**{c}**" for c in overall) + " |")
    w("")

    # --- the decomposition ---------------------------------------------------
    w("## Where the difference comes from")
    w("")
    w("The column that matters is the last one. *Invalid* answers are ones outside the")
    w("declared type -- an invented label, unparseable output. Jev and the constrained")
    w("arm cannot produce them by construction, so any gap in that column is mechanical")
    w("and would shrink if you improved the parser. *Wrong* answers are in-type and")
    w("simply incorrect; that column is the real test of judgement, and it is the one")
    w("to quote.")
    w("")
    w("Decision tasks only: an agentic episode makes several decisions and is")
    w("graded once on its end state, so a per-answer breakdown does not apply to it.")
    w("")
    w("| arm | invalid | wrong (of valid) | correct |")
    w("|---|---|---|---|")
    for arm in arms:
        # One outcome per result here, which is what keeps the three columns
        # summing to the attempt count.
        scoped = [r for r in results
                  if r.arm == arm and by_id.get(r.task_id) is not None
                  and by_id[r.task_id].mode == "decision"]
        attempts = len(scoped)
        if not attempts:
            continue
        invalid = sum(1 for r in scoped if r.violations)
        valid = attempts - invalid
        correct = sum(1 for r in scoped if r.success)
        wrong = valid - correct
        w(f"| {arm} | {invalid}/{attempts} ({invalid / attempts:.1%}) | "
          f"{wrong}/{max(valid,1)} ({wrong / max(valid, 1):.1%}) | "
          f"{correct}/{attempts} ({correct / attempts:.1%}) |")
    w("")
    if "constrained" in arms:
        w("Read the ladder as two steps: **solo -> constrained** is what schema-forcing")
        w("alone buys, and **constrained -> jev** is what Jev's judgement adds on top.")
        w("If the second step is flat, the honest conclusion is that a JSON schema on")
        w("Ornith would have got you the same place without a second vendor.")
    else:
        w("> The `constrained` arm was not run, so these results **cannot** separate")
        w("> Jev's judgement from the effect of merely forcing a schema. Re-run with")
        w("> all three arms before drawing a conclusion.")
    w("")

    # --- safety, both directions --------------------------------------------
    traps = {t.id for t in tasks if t.is_trap}
    if traps:
        w("## Safety")
        w("")
        w("Refusal rate on traps is only half the picture. An arm that refuses")
        w("everything scores perfectly here and is useless in practice, so the")
        w("false-refusal column is shown alongside it. Judge them together.")
        w("")
        w("| arm | traps refused | safe work wrongly refused |")
        w("|---|---|---|")
        for arm in arms:
            trap_results = [r for r in results if r.arm == arm and r.task_id in traps]
            safe_results = [r for r in results if r.arm == arm and r.task_id not in traps]
            caught = sum(1 for r in trap_results if r.success)
            over = sum(1 for r in safe_results if r.escalated)
            w(f"| {arm} | {caught}/{len(trap_results)} "
              f"({caught / max(len(trap_results),1):.0%}) | {over}/{len(safe_results)} "
              f"({over / max(len(safe_results),1):.0%}) |")
        w("")

    # --- calibration ---------------------------------------------------------
    calibrated = [arm for arm in arms if any(
        o.confidence is not None for r in results if r.arm == arm for o in r.outcomes)]
    if calibrated:
        w("## Calibration")
        w("")
        w("Brier score over the full probability vector (lower is better) and expected")
        w("calibration error on the reported confidence. This is what decides whether a")
        w("confidence gate is worth having: gating on uncalibrated confidence discards")
        w("good answers at random.")
        w("")
        w("| arm | Brier | ECE | n |")
        w("|---|---|---|---|")
        for arm in calibrated:
            predictions: list[tuple[dict[str, float], str]] = []
            confidences: list[tuple[float, bool]] = []
            for r in results:
                if r.arm != arm:
                    continue
                task = by_id.get(r.task_id)
                if task is None or task.expected is None:
                    continue
                for o in r.outcomes:
                    if o.probabilities and not o.violation:
                        predictions.append((o.probabilities, task.expected))
                    if o.confidence is not None and not o.violation:
                        confidences.append((o.confidence, o.label == task.expected))
            if predictions or confidences:
                w(f"| {arm} | {brier(predictions):.4f} | {ece(confidences):.4f} | {len(confidences)} |")
        w("")
        w("> Note on nouls: Jev returns a probability for a yes/no question but no")
        w("> confidence field. This harness derives one as `|p - 0.5| * 2`. That is our")
        w("> construction, not a number the API reported, and noul rows in the ECE")
        w("> column should be read with that in mind.")
        w("")

    # --- paired tests --------------------------------------------------------
    if len(arms) > 1:
        w("## Paired comparisons")
        w("")
        w("Exact McNemar on per-task majority verdicts. Exact rather than chi-square")
        w("because discordant counts on suites this size are usually single digits,")
        w("where the approximation misleads.")
        w("")
        w("| comparison | discordant (a wins / b wins) | p |")
        w("|---|---|---|")
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                test = mcnemar(_majority(results, a), _majority(results, b))
                w(f"| {a} vs {b} | {test.b} / {test.c} | {test.p:.4f} |")
        w("")

    # --- cost ----------------------------------------------------------------
    w("## Latency")
    w("")
    w("| arm | median decision | mean | total wall |")
    w("|---|---|---|---|")
    for arm in arms:
        latencies = sorted(o.latency_ms for r in results if r.arm == arm for o in r.outcomes)
        if not latencies:
            continue
        median = latencies[len(latencies) // 2]
        mean = sum(latencies) / len(latencies)
        total = sum(r.latency_ms for r in results if r.arm == arm)
        w(f"| {arm} | {median:.0f} ms | {mean:.0f} ms | {total / 1000:.1f} s |")
    w("")

    # --- failures worth reading ---------------------------------------------
    w("## Failures to look at")
    w("")
    w("Items where the arms disagreed. These are worth more of your time than the")
    w("aggregate tables -- they are where you find out whether a label is wrong, a")
    w("task is ambiguous, or something real is happening.")
    w("")
    disagreements = 0
    for task in tasks:
        verdicts = {arm: _majority(results, arm).get(task.id) for arm in arms}
        present = {k: v for k, v in verdicts.items() if v is not None}
        if len(set(present.values())) > 1:
            disagreements += 1
            if disagreements <= 15:
                marks = ", ".join(f"{k}={'ok' if v else 'X'}" for k, v in present.items())
                w(f"- `{task.id}` ({task.suite}): {marks}"
                  + (f" -- {task.notes}" if task.notes else ""))
    if not disagreements:
        w("_No disagreements: every arm got the same items right and wrong._")
    elif disagreements > 15:
        w(f"- ... and {disagreements - 15} more (see the JSON for all of them)")
    w("")
    return "\n".join(out)


def save_json(path: str, results: list[TaskResult], meta: dict[str, Any]) -> None:
    payload = {
        "meta": meta,
        "results": [
            {**asdict(r), "outcomes": [
                {**asdict(o), "kind": o.kind.value if isinstance(o.kind, Kind) else o.kind}
                for o in r.outcomes
            ]}
            for r in results
        ],
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
