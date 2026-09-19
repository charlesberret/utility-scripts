"""Command line: `job <command>`."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

from bench.config import JevConfig, OrnithConfig, RunConfig
from bench.report import build, save_json
from bench.suites import load
from bench.types import Task, TaskResult


def _deciders(arms: list[str], run: RunConfig, fake: bool = False) -> dict[str, Any]:
    from bench.deciders.constrained import ConstrainedDecider
    from bench.deciders.jev import JevDecider
    from bench.deciders.solo import SoloDecider

    built: dict[str, Any] = {}
    if fake:
        from bench.fakes import FakeJevClient, FakeOrnithClient, fake_question
        from bench.ornith import Ornith

        ornith = Ornith(OrnithConfig(), client=FakeOrnithClient())
        for arm in arms:
            if arm == "solo":
                built[arm] = SoloDecider(ornith=ornith)
            elif arm == "constrained":
                built[arm] = ConstrainedDecider(ornith=ornith, two_step=False)
            elif arm == "jev":
                built[arm] = JevDecider(
                    JevConfig(), client=FakeJevClient(),
                    bool_threshold=run.bool_threshold, question_builder=fake_question,
                )
        return built

    ornith_config = OrnithConfig()
    for arm in arms:
        if arm == "solo":
            built[arm] = SoloDecider(ornith_config)
        elif arm == "constrained":
            built[arm] = ConstrainedDecider(ornith_config)
        elif arm == "jev":
            built[arm] = JevDecider(JevConfig(), bool_threshold=run.bool_threshold)
        else:
            raise SystemExit(f"unknown arm {arm!r}; choose from solo, constrained, jev")
    return built


def _execute(
    tasks: list[Task], deciders: dict[str, Any], run: RunConfig, *, fake: bool = False
) -> list[TaskResult]:
    from bench.runner import run_agentic, run_decision
    from bench.sandbox import Sandbox, docker_available

    decision_tasks = [t for t in tasks if t.mode == "decision"]
    agentic_tasks = [t for t in tasks if t.mode == "agentic"]
    results: list[TaskResult] = []

    total = len(decision_tasks) * len(deciders) * run.repeats
    done = 0
    for arm, decider in deciders.items():
        for repeat in range(run.repeats):
            for task in decision_tasks:
                results.append(run_decision(task, decider, repeat, run.confidence_gate))
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  decisions {done}/{total}", end="\r", file=sys.stderr, flush=True)
    if decision_tasks:
        print(file=sys.stderr)

    if agentic_tasks:
        if fake or not docker_available():
            why = "fake mode" if fake else "docker is not available"
            print(f"  skipping {len(agentic_tasks)} agentic task(s): {why}", file=sys.stderr)
        else:
            from bench.ornith import Ornith

            ornith = Ornith(OrnithConfig())
            for arm, decider in deciders.items():
                for repeat in range(run.repeats):
                    for task in agentic_tasks:
                        print(f"  agentic {arm} r{repeat} {task.id}", file=sys.stderr)
                        try:
                            with Sandbox(run.sandbox_image) as box:
                                results.append(run_agentic(task, decider, ornith, box, repeat, run.confidence_gate))
                        except Exception as exc:
                            results.append(TaskResult(
                                task_id=task.id, suite=task.suite, arm=arm, repeat=repeat,
                                success=False, error=f"{type(exc).__name__}: {exc}",
                            ))
    return results


def _emit(results: list[TaskResult], tasks: list[Task], meta: dict[str, Any], run: RunConfig, label: str) -> None:
    Path(run.results_dir).mkdir(parents=True, exist_ok=True)
    stem = f"{run.results_dir}/{label}"
    save_json(f"{stem}.json", results, meta)
    report = build(results, tasks, meta)
    Path(f"{stem}.md").write_text(report)
    print(report)
    print(f"\n-> {stem}.md and {stem}.json", file=sys.stderr)


def cmd_run(args: argparse.Namespace) -> int:
    run = RunConfig(
        repeats=args.repeats, confidence_gate=args.gate,
        bool_threshold=args.bool_threshold, results_dir=args.out,
    )
    tasks = load(args.suite)
    arms = args.arms.split(",")
    if "constrained" not in arms and "jev" in arms and "solo" in arms:
        print("note: without the `constrained` arm this run cannot separate Jev's "
              "judgement from the effect of any schema. See the report.", file=sys.stderr)
    deciders = _deciders(arms, run, fake=args.fake)
    ornith = OrnithConfig()
    meta = {
        "run_id": uuid.uuid4().hex[:8], "arms": arms, "repeats": run.repeats,
        "ornith_model": "fake" if args.fake else ornith.model,
        "ornith_base_url": "fake" if args.fake else ornith.base_url,
        "jev_model": "fake" if args.fake else JevConfig().model,
        "temperature": ornith.temperature, "confidence_gate": run.confidence_gate,
    }
    print(f"{len(tasks)} tasks x {len(arms)} arms x {run.repeats} repeats", file=sys.stderr)
    results = _execute(tasks, deciders, run, fake=args.fake)
    _emit(results, tasks, meta, run, args.label or f"run-{meta['run_id']}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Whole pipeline, fakes only. No GPU, no key, no network."""
    args.fake = True
    args.arms = "solo,constrained,jev"
    args.repeats = args.repeats or 3
    args.gate = 0.0
    args.bool_threshold = 0.5
    args.out = "results"
    args.label = "smoke"
    args.suite = args.suite or ["suites"]
    return cmd_run(args)


def cmd_gate(args: argparse.Namespace) -> int:
    """Sweep the confidence gate. Only meaningful for an arm with calibrated confidence."""
    tasks = load(args.suite)
    rows = []
    for threshold in [round(x / 20, 2) for x in range(0, 19)]:
        run = RunConfig(repeats=args.repeats, confidence_gate=threshold, results_dir=args.out)
        deciders = _deciders(["jev"], run, fake=args.fake)
        results = _execute(tasks, deciders, run, fake=args.fake)
        acted = [r for r in results if not r.escalated]
        correct = sum(1 for r in acted if r.success)
        rows.append((
            threshold,
            len(acted) / max(len(results), 1),
            correct / max(len(acted), 1),
            sum(1 for r in results if r.success) / max(len(results), 1),
        ))

    print("\n## Confidence gate sweep\n")
    print("Raising the gate trades coverage for precision. The useful threshold is")
    print("the one where precision-when-acting is high enough to trust unattended")
    print("and coverage is high enough to be worth having.\n")
    print("| gate | acted on | precision when acting | overall score |")
    print("|---|---|---|---|")
    for threshold, coverage, precision, overall in rows:
        print(f"| {threshold:.2f} | {coverage:.0%} | {precision:.1%} | {overall:.1%} |")
    return 0


def cmd_tasks(args: argparse.Namespace) -> int:
    """Load and validate every suite without calling a model."""
    from bench.stats import min_detectable_effect

    tasks = load(args.suite)
    by_suite: dict[str, list[Task]] = {}
    for task in tasks:
        by_suite.setdefault(task.suite, []).append(task)
    print(f"{len(tasks)} tasks in {len(by_suite)} suites\n")
    for suite, items in sorted(by_suite.items()):
        traps = sum(1 for t in items if t.is_trap)
        boundary = sum(1 for t in items if "boundary" in t.tags)
        print(f"  {suite:24s} {len(items):3d} tasks  ({traps} traps, {boundary} boundary)")
    print(f"\nA suite of {len(tasks)} can resolve a difference of about "
          f"{min_detectable_effect(len(tasks)):.0%}. Add tasks if you need finer.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check every backend before a long run depends on it."""
    ok = True

    print("Ornith:")
    ornith_config = OrnithConfig()
    print(f"  endpoint  {ornith_config.base_url}")
    print(f"  model     {ornith_config.model}")
    try:
        from bench.ornith import Ornith

        reply = Ornith(ornith_config).chat(
            [{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=200
        )
        print(f"  status    reachable ({reply.latency_ms:.0f} ms), said {reply.text.strip()[:40]!r}")
        if reply.reasoning is None:
            print("  note      no reasoning_content field. Serve with --reasoning-parser qwen3")
            print("            so the <think> block does not land in the answer.")
    except Exception as exc:
        ok = False
        print(f"  status    UNREACHABLE: {type(exc).__name__}: {exc}")

    print("\nJev:")
    jev_config = JevConfig()
    if not jev_config.api_key:
        ok = False
        print("  status    no TYPESAFE_API_KEY set (get one at console.typesafe.ai)")
    else:
        try:
            from bench.deciders.jev import JevDecider
            from bench.types import Kind, Question

            probe = Question(
                name="probe", kind=Kind.CHOICE, instructions="Which colour is named?",
                options={"red": None, "blue": None},
            )
            outcome = JevDecider(jev_config).decide("The sky is blue today.", probe)
            if outcome.error:
                ok = False
                print(f"  status    ERROR: {outcome.error}")
            else:
                print(f"  model     {jev_config.model}")
                print(f"  status    reachable ({outcome.latency_ms:.0f} ms), "
                      f"said {outcome.label!r} at confidence {outcome.confidence:.2f}")
        except ImportError:
            ok = False
            print("  status    typesafe-sdk not installed (uv sync)")

    print("\nSandbox:")
    from bench.sandbox import docker_available

    if docker_available():
        print("  status    docker ready")
    else:
        print("  status    docker unavailable -- decision suites still run, agentic ones skip")

    print("\n" + ("All backends ready." if ok else "Some backends are not ready (see above)."))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="job", description="Does Jev's strictness make Ornith better at real work?"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--suite", action="append", help="suite file or directory (repeatable)")
        p.add_argument("--repeats", type=int, default=5)
        p.add_argument("--out", default="results")
        p.add_argument("--fake", action="store_true", help="use scripted backends, no network")

    run_parser = sub.add_parser("run", help="run a sweep")
    common(run_parser)
    run_parser.add_argument("--arms", default="solo,constrained,jev")
    run_parser.add_argument("--gate", type=float, default=0.0, help="escalate below this confidence")
    run_parser.add_argument("--bool-threshold", type=float, default=0.5, dest="bool_threshold")
    run_parser.add_argument("--label")
    run_parser.set_defaults(func=cmd_run)

    smoke_parser = sub.add_parser("smoke", help="end-to-end on fakes; no GPU, key or network")
    smoke_parser.add_argument("--suite", action="append")
    smoke_parser.add_argument("--repeats", type=int, default=3)
    smoke_parser.set_defaults(func=cmd_smoke)

    gate_parser = sub.add_parser("sweep-gate", help="sweep the confidence gate threshold")
    common(gate_parser)
    gate_parser.set_defaults(func=cmd_gate)

    tasks_parser = sub.add_parser("tasks", help="validate suites and show their size")
    tasks_parser.add_argument("--suite", action="append")
    tasks_parser.set_defaults(func=cmd_tasks)

    doctor_parser = sub.add_parser("doctor", help="check Ornith, Jev and docker")
    doctor_parser.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
