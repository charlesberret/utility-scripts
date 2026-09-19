"""Arm 2: Ornith with decoding constrained to a JSON schema. The control.

This arm shares its model, prompt and sampling with `solo` and differs only in
that the grammar forbids an out-of-type answer. So:

  solo -> constrained   measures what schema-forcing alone buys you.
  constrained -> jev    measures what Jev's judgement buys you on top.

Reporting only solo vs jev would confound the two and overstate Jev. This arm
is what turns the run into an experiment rather than a demo.

A caveat worth keeping in view when reading results: constrained decoding
interacts badly with reasoning models. If the grammar is applied from the
first token, the model cannot emit its <think> block and loses the reasoning
that Ornith was trained to use. We therefore run this arm in two steps --
think freely, then re-ask under the grammar -- so that the comparison is about
the constraint and not about lobotomising the model. Servers that cannot do
constrained decoding at all will silently return unconstrained text; the
violation counter catches that, and a non-zero violation count in this arm
means the flag did not take and the run should be discarded.
"""

from __future__ import annotations

import json
from typing import Any

from bench.config import OrnithConfig
from bench.deciders.solo import _prompt
from bench.ornith import Ornith
from bench.types import Kind, Outcome, Question


def schema_for(question: Question) -> dict[str, Any]:
    """The JSON schema pinning the answer to the declared type."""
    if question.kind is Kind.CHOICE:
        answer: dict[str, Any] = {"type": "string", "enum": question.labels}
    elif question.kind is Kind.NOUL:
        answer = {"type": "boolean"}
    else:
        assert question.levels is not None
        answer = {"type": "integer", "minimum": 0, "maximum": len(question.levels) - 1}
    return {
        "type": "object",
        "properties": {"answer": answer},
        "required": ["answer"],
        "additionalProperties": False,
    }


class ConstrainedDecider:
    name = "constrained"

    def __init__(
        self,
        config: OrnithConfig | None = None,
        ornith: Ornith | None = None,
        *,
        two_step: bool = True,
    ) -> None:
        self.ornith = ornith or Ornith(config)
        self.two_step = two_step

    def decide(self, state: str, question: Question) -> Outcome:
        prompt = f"{state}\n\n---\n\n{_prompt(question)}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You answer classification questions."},
            {"role": "user", "content": prompt},
        ]
        total_latency = 0.0
        raw_parts: list[str] = []

        try:
            if self.two_step:
                # Let it reason without the grammar, then bind the answer.
                free = self.ornith.chat(messages, max_tokens=1024)
                total_latency += free.latency_ms
                raw_parts.append(free.text)
                messages.append({"role": "assistant", "content": free.text or "(no answer)"})
                messages.append({"role": "user", "content": "Return that answer as JSON matching the schema."})

            reply = self.ornith.chat(messages, json_schema=schema_for(question), max_tokens=256)
            total_latency += reply.latency_ms
            raw_parts.append(reply.text)
        except Exception as exc:
            return Outcome(
                question=question.name, kind=question.kind, value=None,
                violation=True, error=f"{type(exc).__name__}: {exc}",
            )

        out = Outcome(
            question=question.name, kind=question.kind, value=None,
            latency_ms=total_latency, raw=" || ".join(p for p in raw_parts if p),
        )
        try:
            answer = json.loads(reply.text)["answer"]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Constrained decoding did not take -- the server ignored the flag.
            out.violation = True
            out.error = "constrained decoding produced non-conforming output"
            return out

        if question.kind is Kind.NOUL:
            out.value = bool(answer)
        elif question.kind is Kind.SCORE:
            assert question.levels is not None
            value = float(answer)
            if not 0 <= value <= len(question.levels) - 1:
                out.violation = True
                return out
            out.value = value
        else:
            if str(answer) not in question.labels:
                out.violation = True
                return out
            out.value = str(answer)
        return out
