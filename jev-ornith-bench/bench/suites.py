"""Loading task suites from YAML.

Tasks are data, not code, so that adding a case is a few lines in a file and
needs no Python. That matters more than it sounds: the power table in the
README says a suite needs to be in the low hundreds before it can resolve a
realistic effect, and nobody writes two hundred of anything that requires
touching a module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from bench.types import Kind, Question, Task


def _question(raw: dict[str, Any], task_id: str) -> Question:
    try:
        kind = Kind(raw["kind"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{task_id}: question needs a kind of noul, choice or score") from exc

    options = None
    if kind is Kind.CHOICE:
        given = raw.get("options")
        if isinstance(given, list):
            options = {str(o): None for o in given}
        elif isinstance(given, dict):
            options = {str(k): (str(v) if v is not None else None) for k, v in given.items()}
        else:
            raise ValueError(f"{task_id}: choice question needs options")

    return Question(
        name=raw.get("name", "decision"),
        kind=kind,
        instructions=raw.get("instructions", ""),
        options=options,
        levels=[str(level) for level in raw["levels"]] if raw.get("levels") else None,
        true_means=raw.get("true_means"),
        false_means=raw.get("false_means"),
    )


def load_file(path: Path) -> list[Task]:
    document = yaml.safe_load(path.read_text()) or {}
    suite = document.get("suite", path.stem)
    defaults = document.get("defaults", {}) or {}
    tasks: list[Task] = []

    for index, raw in enumerate(document.get("tasks", [])):
        merged: dict[str, Any] = {**defaults, **raw}
        task_id = str(merged.get("id") or f"{suite}-{index:03d}")
        mode = merged.get("mode", "decision")

        question = None
        if mode == "decision":
            if "question" not in merged:
                raise ValueError(f"{task_id}: decision task needs a question")
            question = _question(merged["question"], task_id)

        expected = merged.get("expected")
        task = Task(
            id=task_id,
            suite=suite,
            mode=mode,
            state=str(merged.get("state", "")),
            question=question,
            expected=None if expected is None else str(expected),
            acceptable=[str(a) for a in merged.get("acceptable", [])],
            goal=merged.get("goal"),
            setup=merged.get("setup"),
            check=merged.get("check"),
            tools=[str(t) for t in merged.get("tools", [])],
            max_turns=int(merged.get("max_turns", 8)),
            is_trap=bool(merged.get("is_trap", False)),
            notes=str(merged.get("notes", "")),
            tags=[str(t) for t in merged.get("tags", [])],
        )

        # Catch a typo'd expected answer at load time. Left undetected it
        # looks exactly like every arm failing that item, which is a
        # genuinely confusing thing to debug halfway through a sweep.
        if task.mode == "decision" and question is not None and task.expected is not None:
            allowed = set(question.labels)
            unknown = {task.expected, *task.acceptable} - allowed
            if unknown:
                raise ValueError(
                    f"{task_id}: expected/acceptable {sorted(unknown)} not among "
                    f"the question's labels {sorted(allowed)}"
                )
        tasks.append(task)
    return tasks


def load(paths: list[str] | None = None, root: str = "suites") -> list[Task]:
    """Load every suite under `root`, or the named files or directories."""
    targets: list[Path] = []
    for item in paths or [root]:
        path = Path(item)
        if path.is_dir():
            targets.extend(sorted(path.rglob("*.yaml")))
        elif path.exists():
            targets.append(path)
        else:
            raise FileNotFoundError(item)

    tasks: list[Task] = []
    seen: set[str] = set()
    for target in targets:
        for task in load_file(target):
            if task.id in seen:
                raise ValueError(f"duplicate task id {task.id!r} in {target}")
            seen.add(task.id)
            tasks.append(task)
    return tasks
