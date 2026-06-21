# Otto ARC-AGI Smoke Harness

This is a minimal reproduction of the GEPA ARC-AGI pipeline described in the
`optimize_anything` blog. It optimizes `seed_agent.py`, a tiny ARC solver
program, against a small ARC slice using `databricks-gemini-3-flash` through the
Databricks `oss` profile.

## Setup

Install optional dependencies only inside this experiment:

```bash
cd examples/otto/experiments/arc_agi
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The runner loads Databricks credentials from the `oss` profile by default. You
can also export them explicitly. No keys are stored in this repo.

```bash
export DATABRICKS_HOST=<oss workspace host>
export DATABRICKS_TOKEN=<oss token>
```

## Cheap dry run

This checks imports, downloads/loads a small ARC slice, and exits before LLM
calls:

```bash
python run.py --dry-run
```

## Tiny smoke optimization

The default budget is intentionally small: 4 train tasks, 4 validation tasks,
`max_metric_calls=10`, `max_workers=2`, and up to 4 model calls per ARC task.

```bash
python run.py
```

Use `--profile <name>` to select a different Databricks CLI profile.

Outputs are written under `outputs/smoke/`, including `best_agent.py`.

## Scaling up

Increase the same flags to approach the full reported ARC run:

```bash
python run.py \
  --train-size 200 \
  --val-size 200 \
  --max-metric-calls 3000 \
  --max-workers 64 \
  --max-llm-calls 10 \
  --run-dir outputs/full
```

The blog reports Gemini 3 Flash improving from 32.5% to 89.5% on the public ARC
test set with `max_metric_calls=3000` and `max_workers=64`. This harness
reproduces the pipeline and defaults to a cheap smoke budget; the full run is
expensive and should be launched only when you intend to spend that budget.
