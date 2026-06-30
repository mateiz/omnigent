# GEPA `optimize_anything` — Blackbox Optimization (Databricks GPT-5 proposer)

End-to-end run of the upstream GEPA **blackbox** example
(`gepa/examples/blackbox/main.py`, "Match or Outperform Optuna at Blackbox
Mathematical Optimization" from the *Introducing optimize_anything* blog) with
**one change**: the reflection/proposer LM is **GPT-5 served through the
Databricks model gateway**, not the public OpenAI API.

This is a **Single-Task Search**: no dataset, just a natural-language objective.
GEPA evolves a Python `solve()` function that minimizes a blackbox objective
within a fixed evaluation budget. **No LLM is in the objective-evaluation loop** —
the LM is only the GEPA proposer; objective evals are sub-second numeric calls.

## Provenance

- **Real upstream code**, vendored from `scratch/gepa/examples/blackbox/`:
  - `evalset/` — the SigOpt EvalSet benchmark functions (`problems.py`,
    `evalset.py`, `LICENSE`), unchanged except a localized sibling import.
  - `blackbox_lib.py` — logic-preserving copy of upstream `utils.py`
    (`BACKGROUND`, `BudgetTracker`, `execute_code`, `extract_best_xs`,
    `SEED_CODE` lives in `run.py`). The only substantive addition is wiring
    `actual_call_count` (referenced by upstream `BudgetTracker.record` but never
    set in the vendored snippet) by counting real `objective_function` calls.
- **Reconstructed wiring** (`run.py`): the upstream `main()` driver plus the
  Databricks-gateway reflection LM. Config values are taken verbatim from
  upstream `main.py`.

## Config (verbatim from upstream `main.py`)

| Setting | Value |
|---|---|
| `problem_index` | `46` = **Sargan**, dim 5 (upstream default) |
| `NUM_PROPOSALS` (budget tracker) | `10` |
| `EVALUATION_BUDGET` | `2000` sub-second objective evals |
| `max_candidate_proposals` | `20` |
| `cache_evaluation` | `True` |
| seed program | minimal uniform-random single-sample solver |
| proposer / `reflection_lm` | `openai/gpt-5` → **`databricks/databricks-gpt-5`** |

## Model wiring

`databricks-gpt-5` is a READY serving endpoint on the `oss` Databricks profile
(`https://ai-oss-ecosystem-integration-testing.cloud.databricks.com`). It maps
to the real `gpt-5-2025-08-07`. The `oss` profile authenticates via OAuth
(`databricks-cli`), so `run.py` mints a short-lived bearer token from
`Config(profile="oss").authenticate()` and points LiteLLM's databricks provider
at `{host}/serving-endpoints`. LiteLLM does not meter cost for this endpoint, so
spend is estimated at GPT-5 public list pricing ($1.25 / $10 per 1M in/out).

## Run

```bash
PY=../arc_agi/.venv/bin/python   # has gepa, databricks-sdk, litellm, numpy, scipy, scikit-learn
$PY -u run.py --problem-index 46 --run-dir outputs/p46_full
```

## Result (single clean run, 2026-06-30)

| Metric | Value |
|---|---|
| Baseline (seed) objective | **151.25** |
| Best evolved objective (GEPA-measured) | **−100.0** (improvement 251.25) |
| Proposal / reflection LM calls | 20 |
| Objective evals used | 1989 / 2000 |
| GPT-5 spend (USD, estimated) | **$1.1194** |
| Tokens in / out | 406,488 / 61,125 |
| Wallclock | 727.7 s (~12.1 min) |

The evolved solver (`outputs/p46_full/best_program.py`) is a 200+-line hybrid
optimizer (GP/Matérn surrogate + UCB, trust-region local search, LHS + farthest-
point multi-start, density-aware sampling) — discovered entirely by GEPA. It is
stochastic (unseeded RNG) and reproducibly lands around −80 to −100 using only
200 objective evals, beating brute random search (~−69 over 20,000 samples).
