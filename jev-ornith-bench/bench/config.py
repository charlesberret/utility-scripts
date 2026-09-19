"""Run configuration, read from the environment with sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is a convenience, not a requirement
    pass


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class OrnithConfig:
    """Points at any OpenAI-compatible server: vLLM, SGLang, llama.cpp, Ollama."""

    base_url: str = field(default_factory=lambda: os.environ.get("ORNITH_BASE_URL", "http://localhost:8000/v1"))
    api_key: str = field(default_factory=lambda: os.environ.get("ORNITH_API_KEY", "EMPTY"))
    model: str = field(default_factory=lambda: os.environ.get("ORNITH_MODEL", "Ornith-1.5-9B"))
    # Model-card recommendation for general use. The published Terminal-Bench
    # and SWE-bench numbers used temperature=1.0; if you are trying to
    # reproduce those, set it here rather than in the code.
    temperature: float = field(default_factory=lambda: _f("ORNITH_TEMPERATURE", 0.6))
    top_p: float = field(default_factory=lambda: _f("ORNITH_TOP_P", 0.95))
    top_k: int = field(default_factory=lambda: _i("ORNITH_TOP_K", 20))
    max_tokens: int = field(default_factory=lambda: _i("ORNITH_MAX_TOKENS", 2048))
    timeout: float = field(default_factory=lambda: _f("ORNITH_TIMEOUT", 120.0))


@dataclass
class JevConfig:
    """The SDK reads TYPESAFE_API_KEY itself; these mirror its own env names."""

    api_key: str | None = field(default_factory=lambda: os.environ.get("TYPESAFE_API_KEY"))
    base_url: str | None = field(default_factory=lambda: os.environ.get("TYPESAFE_BASE_URL"))
    model: str = field(default_factory=lambda: os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest"))
    timeout: float = field(default_factory=lambda: _f("TYPESAFE_TIMEOUT", 10.0))


@dataclass
class RunConfig:
    """How a sweep is executed.

    `repeats` defaults to 5 because a 9B model at temperature 0.6 is noisy
    enough that a single pass routinely swings several points on a
    fifty-item suite. One repeat produces a number; it does not produce a
    finding.
    """

    repeats: int = 5
    seed: int = 0
    # Below this confidence the gated arm escalates instead of acting. Jev's
    # calibrated confidence is the whole reason this knob can exist; sweep it
    # with `job sweep-gate` rather than guessing.
    confidence_gate: float = 0.0
    bool_threshold: float = 0.5
    concurrency: int = 4
    sandbox_image: str = field(default_factory=lambda: os.environ.get("BENCH_SANDBOX_IMAGE", "debian:bookworm-slim"))
    results_dir: str = "results"
