---
name: optimize-anything
description: Scaffold and run a GEPA optimize_anything experiment over a text artifact with optional per-experiment dependencies.
---

# optimize-anything — reflective artifact optimization

Use this skill when the user wants to improve a prompt, program, agent
architecture, policy, or other text artifact with GEPA's
`optimize_anything` loop.

## Setup

Keep dependencies local to the experiment, not in Omnigent's package deps:

```bash
cd <experiment-dir>
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

For GEPA experiments that use ARC or LLM calls, `requirements.txt` usually
contains only optional experiment deps such as:

```text
gepa
datasets
litellm
```

## Core API

Import and call:

```python
from gepa.optimize_anything import optimize_anything

result = optimize_anything(
    seed_candidate=seed_candidate,
    evaluator=evaluator,
    dataset=dataset,
    valset=valset,
    objective=objective,
    background=background,
    config=config,
)
```

Arguments:
- `seed_candidate`: the starting text artifact, or `None` for seedless mode.
- `evaluator`: callable that runs one candidate on one example and returns a
  numeric score plus trace/side information for reflection.
- `dataset`: training examples used during optimization.
- `valset`: held-out examples used to select the best candidate.
- `objective`: concise description of what score should improve.
- `background`: domain context, artifact contract, constraints, and scoring
  semantics.
- `config`: GEPA configuration, including run directory, metric-call budget,
  parallelism, caching, and the reflection LLM.

## Model access

When evaluator code makes LLM calls, route them through Databricks-served models
on the `oss` profile. The Otto examples use `litellm` with:

```python
completion(
    model="databricks/databricks-gemini-3-flash",
    custom_llm_provider="databricks",
    api_base=os.environ.get("DATABRICKS_HOST"),
    api_key=os.environ.get("DATABRICKS_TOKEN"),
    messages=[{"role": "user", "content": prompt}],
)
```

Load credentials from the configured `oss` profile before running, for example
by exporting `DATABRICKS_HOST` and `DATABRICKS_TOKEN` from that profile. Never
hard-code provider keys in experiment files.

## Candidate shape

Define the experiment contract before optimizing:
- Keep the seed artifact small and runnable.
- Make the evaluator deterministic except for intended LLM calls.
- Include enough side information for GEPA to understand failures.
- Fix the train/validation split before comparing candidates.
- Save the best candidate and scores under an ignored output directory.

## Budget

Scale GEPA with `config`, especially:
- `max_metric_calls`: total evaluator-call budget.
- `max_workers`: parallel evaluator workers.
- `run_dir`: output/log directory.
- `cache_evaluation`: reuse identical candidate/example scores.

Start tiny. The ARC-AGI harness under `examples/otto/experiments/arc_agi/`
defaults to about 4 train tasks, 4 validation tasks, `max_metric_calls=10`, and
`max_workers=2`. The full ARC-AGI result reported in the blog uses
`max_metric_calls=3000` and `max_workers=64`, improving Gemini 3 Flash from
32.5% to 89.5% on the public ARC test set.
