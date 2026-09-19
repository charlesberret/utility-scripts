"""Statistics for small, noisy, paired runs.

Three facts about this experiment drive everything here:

1. A 9B model at temperature 0.6 is genuinely stochastic, so every task is run
   several times. Those repeats are *not* independent observations -- they are
   clustered within a task. Treating them as independent would shrink the
   error bars by roughly sqrt(repeats) and manufacture significance out of
   nothing. The bootstrap below therefore resamples *tasks*, carrying each
   task's repeats along with it.

2. All arms see the same tasks, so comparisons are paired. A paired test is
   far more powerful here than comparing two independent proportions, which
   matters when a suite is fifty items long.

3. With suites this small, a difference of a few points is usually noise. The
   report prints intervals rather than bare means for that reason.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.1%} [{self.lo:.1%}, {self.hi:.1%}]"


def cluster_bootstrap(
    by_task: dict[str, list[bool]],
    *,
    resamples: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile CI for a mean rate, resampling tasks rather than repeats.

    `by_task` maps a task id to that task's per-repeat outcomes.
    """
    task_ids = list(by_task)
    if not task_ids:
        return Interval(float("nan"), float("nan"), float("nan"), 0)

    def rate(ids: list[str]) -> float:
        hits = total = 0
        for tid in ids:
            results = by_task[tid]
            hits += sum(1 for r in results if r)
            total += len(results)
        return hits / total if total else float("nan")

    point = rate(task_ids)
    if len(task_ids) == 1:
        return Interval(point, float("nan"), float("nan"), sum(len(v) for v in by_task.values()))

    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [task_ids[rng.randrange(len(task_ids))] for _ in task_ids]
        value = rate(sample)
        if not math.isnan(value):
            draws.append(value)
    draws.sort()
    if not draws:
        return Interval(point, float("nan"), float("nan"), 0)
    lo = draws[int((alpha / 2) * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return Interval(point, lo, hi, sum(len(v) for v in by_task.values()))


def _binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial p-value for b successes in b+c trials at p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


@dataclass
class Mcnemar:
    b: int  # a right, b wrong
    c: int  # a wrong, b right
    p: float

    def __str__(self) -> str:
        return f"discordant {self.b}/{self.c}, p={self.p:.4f}"


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> Mcnemar:
    """Exact McNemar over items both arms attempted.

    Uses the exact binomial rather than the chi-square approximation because
    discordant counts here are routinely in single digits, where the
    approximation is unreliable.
    """
    shared = a.keys() & b.keys()
    only_a = sum(1 for k in shared if a[k] and not b[k])
    only_b = sum(1 for k in shared if not a[k] and b[k])
    return Mcnemar(only_a, only_b, _binom_two_sided(only_a, only_b))


def brier(predictions: list[tuple[dict[str, float], str]]) -> float:
    """Multi-class Brier score. Lower is better; 0 is perfect.

    Each entry is (probabilities by label, the true label). This is the direct
    test of Jev's calibration claim, and it is only computable for arms that
    return probabilities at all.
    """
    if not predictions:
        return float("nan")
    total = 0.0
    for probabilities, truth in predictions:
        labels = set(probabilities) | {truth}
        total += sum((probabilities.get(label, 0.0) - (1.0 if label == truth else 0.0)) ** 2 for label in labels)
    return total / len(predictions)


def ece(predictions: list[tuple[float, bool]], bins: int = 10) -> float:
    """Expected calibration error over (confidence, was_correct) pairs.

    Answers "when this arm says 0.9, is it right 90% of the time?" -- which is
    what makes a confidence gate trustworthy. A gate built on uncalibrated
    confidence just throws away good answers at random.
    """
    if not predictions:
        return float("nan")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, correct in predictions:
        index = min(bins - 1, max(0, int(confidence * bins)))
        buckets[index].append((confidence, correct))
    total = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_conf = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, ok in bucket if ok) / len(bucket)
        total += (len(bucket) / len(predictions)) * abs(mean_conf - accuracy)
    return total


def min_detectable_effect(n_tasks: int, baseline: float = 0.6) -> float:
    """Roughly the smallest accuracy gain a suite this size can resolve.

    Note what is absent: `repeats` does not appear. Repeats sharpen the
    estimate of how a given task behaves, but the unit of independent
    information is the task, so adding repeats does not buy resolving power
    the way adding tasks does. If the effect you care about is small, write
    more tasks -- running the same twelve items twenty times will not help.

    Deliberately crude, and deliberately printed before a run: the most
    common way to waste a GPU afternoon here is to build a twelve-item suite
    and then read a six-point swing as a discovery.
    """
    n = max(1, n_tasks)
    se = math.sqrt(2 * baseline * (1 - baseline) / n)
    return 2.8 * se  # ~80% power at alpha=0.05, two-sided
