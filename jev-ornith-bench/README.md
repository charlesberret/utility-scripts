# Does Jev's strictness make Ornith more capable?

A harness for one question: if you put [Jev](https://typesafe.ai) — TypeSafe's
System One model, which selects from typed options and cannot return anything
outside them — in front of a local model like
[Ornith](https://huggingface.co/ornith-ai), does the pair get measurably
better at real sysadmin and clerical work than the local model alone?

Everything here is built to answer that without fooling us.

## The design, and the one thing that makes it an experiment

Ornith does all the **generation** in every arm: it writes the shell command,
reads the log, drafts the next step. What changes between arms is only who
answers the typed **control-flow** questions — which action, is this safe to
run unattended, are we done, how urgent is this.

| arm | who decides | can emit an invalid answer? |
|---|---|---|
| `solo` | Ornith, prompted in prose, reply parsed | **yes** |
| `constrained` | Ornith, decoding pinned to a JSON schema | no |
| `jev` | Jev answers the typed question directly | no |

The middle arm is the point. Comparing only `solo` to `jev` conflates two
completely different effects:

1. Jev cannot emit an invalid answer, so every malformed reply from the solo
   arm becomes a free win — an artifact of prompt-and-parse that a better
   parser would erase; and
2. Jev might genuinely make better judgements.

Only the second is interesting, and only the second is a reason to add a
second vendor to your stack. So read the ladder as two steps:

- **solo → constrained** — what schema-forcing alone buys you.
- **constrained → jev** — what Jev's judgement adds on top.

If that second step is flat, the honest conclusion is that a JSON schema on
Ornith would have got you to the same place for free. The report says so in
those words.

## What Jev can and cannot be asked

Jev has three primitives and no others: **noul** (yes/no, as a calibrated
probability), **choice** (one label from a closed set), **score** (a rating
against an ordered rubric). It generates no text at all.

That shapes the whole benchmark. "Extract the vendor name from this invoice"
is not a Jev task and never will be — it is a string. "Which of these twelve
known vendors is this invoice from" is. The clerical suites are therefore
built entirely from *decisions*, not transcription, and the honest framing of
the hypothesis is narrower than it first sounds:

> Strictness should help with the parts of clerical and sysadmin work that are
> classification, routing, prioritisation and go/no-go judgement. It cannot
> help with the parts that are writing things down.

If the result is positive, that is the claim it supports.

## Running it

```bash
uv sync
cp .env.example .env    # add TYPESAFE_API_KEY, point ORNITH_BASE_URL at your server
```

Check the whole thing works before spending GPU time on it — this needs no
GPU, no API key and no network, and exercises every stage including the
statistics and the report:

```bash
job smoke
```

Then the real backends:

```bash
./scripts/serve-ornith.sh vllm     # in another terminal
job doctor                          # confirms Ornith, Jev and docker
job tasks                           # validates suites, prints what they can resolve
job run --repeats 5
job run --suite suites/sysadmin --arms solo,constrained,jev --repeats 10
job sweep-gate                      # where to set the confidence gate
```

Results land in `results/` as a markdown report plus the full JSON.

## Reading the output without fooling yourself

**Check the power line first.** It is printed above every result table. With
55 tasks you can resolve a difference of roughly 26 percentage points — which
is to say, almost nothing you actually care about. The shipped suites are a
*seed*, sized to prove the harness works, not to settle the question. Getting
to a trustworthy answer means writing tasks until each suite is in the low
hundreds. That is the real work this framework exists to make cheap: tasks are
YAML, so adding one is four lines and no Python.

**Repeats do not substitute for tasks.** Ornith at temperature 0.6 is
stochastic, so every task runs several times — but repeats within a task are
correlated, and the unit of independent information is the task. The bootstrap
resamples tasks rather than repeats for exactly this reason; treating repeats
as independent would shrink every interval by roughly √(repeats) and
manufacture significance out of nothing.

**Read both safety columns together.** An arm that refuses everything gets a
perfect trap-refusal score and is useless. The report always shows
false-refusals alongside.

**Watch the `constrained` arm's violation count.** It should be exactly zero.
Anything else means your server ignored the schema flag, that arm silently
degraded into `solo`, and the run should be thrown away.

**Go read the disagreements.** The last section of the report lists items
where the arms split. That list is worth more of your attention than any
aggregate — it is where you discover a mislabelled task, a genuinely ambiguous
one, or something real.

## Layout

```
bench/
  types.py       Question / Outcome / Task — the shared vocabulary
  deciders/      the three arms; solo.py's parser is deliberately generous
  ornith.py      OpenAI-compatible client, strips <think> defensively
  runner.py      decision mode and the agentic loop
  sandbox.py     throwaway docker container for the agentic tasks
  stats.py       cluster bootstrap, exact McNemar, Brier, ECE
  report.py      the decomposition tables
  fakes.py       scripted backends so everything runs offline
suites/          tasks as YAML: sysadmin/ and clerical/
```

## Notes and caveats

- **Confidence gating** (`--gate`) is where Jev should earn its place: act when
  confident, escalate when not. It only works if the confidence is calibrated,
  which is why the report scores Brier and ECE. `job sweep-gate` finds the
  threshold; don't guess it.
- **Nouls have no confidence field.** Jev returns a probability for a yes/no
  question but no separate confidence, so this harness derives one as
  `|p − 0.5| × 2`. That is our construction, not a number the API reported,
  and it is flagged wherever it appears.
- **We call `typesafe_sdk` directly**, not the higher-level `jev` decorator
  package. The decorator is pleasanter to write but discards the
  probabilities, and its own docs say to drop to the SDK when you need
  confidence — which is precisely the signal under test here.
- **Agentic tasks need docker** and run with networking off. Traps in that
  suite give the model genuine capability to do damage inside a boundary that
  makes the damage safe; blocking the action at the tool layer would score the
  harness rather than the model.
- **Sampling defaults** are the model card's recommendation (0.6 / 0.95 / 20).
  Ornith's published Terminal-Bench and SWE-bench figures used temperature
  1.0; set `ORNITH_TEMPERATURE=1.0` if you are trying to line up with those.
