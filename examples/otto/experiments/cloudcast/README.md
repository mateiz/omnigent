# GEPA `optimize_anything` — CloudCast Broadcast Routing (Databricks Gemini 3 Pro proposer)

End-to-end run of the upstream GEPA **CloudCast** example
(`gepa/examples/adrs/cloudcast/main.py`, section *"Discover Cloud Algorithms That
Cut Costs up to 40%"* from the *Introducing optimize_anything* blog) with
**one change**: the reflection/proposer LM is **Gemini 3 Pro served through the
Databricks model gateway**, not the public Gemini API.

This is a **Generalization** run: `dataset == valset ==` the 5 multi-cloud
broadcast configs. GEPA evolves a Python `search_algorithm(src, dsts, G,
num_partitions)` that returns a `BroadCastTopology`, scored by a pure Python
cloud simulator. **No LLM is in the objective-evaluation loop** — the LM is only
the GEPA proposer; objective evals are near-free simulator runs.

## Provenance

- **Real upstream code**, vendored from `scratch/gepa/examples/adrs/cloudcast/`:
  - `utils/cloudcast/` — the broadcast simulator + graph library (`simulator.py`,
    `broadcast.py`, `utils.py`, `initial_program.py`, `config/`, `profiles/`),
    unchanged except localized package imports
    (`examples.adrs.cloudcast.utils.cloudcast` → `utils.cloudcast`).
  - `utils/dataset.py`, `utils/simulation.py`, `utils/wandb_auth.py`, `utils/lm.py`
    — copied verbatim (only `simulation.py` had the same import localized).
- **Reconstructed wiring** (`run.py`): the upstream `main()` driver plus the
  Databricks-gateway reflection LM, a per-call cost tracker, and a hard USD
  budget guard. `INITIAL_PROGRAM`, `OPTIMIZATION_OBJECTIVE`,
  `OPTIMIZATION_BACKGROUND`, and the `evaluate()` body are copied verbatim from
  upstream `main.py`; `GEPAConfig` matches `main.py` field-for-field (wandb off).

## Config (verbatim from upstream `main.py`; blog snippet cited)

| Setting | Value | Source |
|---|---|---|
| proposer / `reflection_lm` | `gemini-3-pro-preview` → **`databricks/databricks-gemini-3-pro`** | blog snippet `appendix-d-cloudcast.snippet`; `main.py:--model` |
| `max_metric_calls` | `100` | `main.py` default; blog `EngineConfig(max_metric_calls=100)` |
| `dataset` = `valset` | 5 configs: `intra_aws, intra_azure, intra_gcp, inter_agz, inter_gaz2` | `utils/dataset.py:_CONFIG_FILES`; blog snippet |
| `reflection_minibatch_size` | `3` | `main.py` default |
| `skip_perfect_score` | `False` | `main.py` |
| `seed` | `0` | `main.py` |
| `num_vms` | `2` | `utils/dataset.py` |
| seed program | baseline Dijkstra shortest-cost single-path router | `main.py:INITIAL_PROGRAM` |
| objective score | `1 / (1 + total_cost)` (higher is better) | `main.py:evaluate` |

The blog snippet shows a minimal `GEPAConfig(engine=EngineConfig(max_metric_calls=100),
reflection=ReflectionConfig(reflection_lm=make_reflection_lm("gemini-3-pro-preview")))`.
The vendored `main.py` is the fuller upstream form of the same config (adds
`seed=0`, `track_best_outputs`, `reflection_minibatch_size=3`,
`skip_perfect_score=False`); we follow `main.py`.

## Model wiring

`databricks-gemini-3-pro` is a READY serving endpoint on the `oss` Databricks
profile (`https://ai-oss-ecosystem-integration-testing.cloud.databricks.com`); it
serves the real `gemini-3.1-pro-preview` (reasoning model). Because the `oss`
profile shares its host with another profile, the SDK's hostname-based
`Config.authenticate()` is ambiguous, so `run.py` mints a short-lived OAuth
bearer token by shelling out to `databricks auth token -p oss` and points
LiteLLM's `databricks` provider at `{host}/serving-endpoints`.

**Cost tracking:** LiteLLM has no price map for this endpoint (metered cost = 0),
so spend is **estimated at Gemini 3 Pro list pricing ($2 / $12 per 1M in/out**,
standard <200K-token tier). Output tokens are counted as `total_tokens −
prompt_tokens` so billed *reasoning* tokens are included. Per call:
`cost = max(metered, estimated)`.

**Hard budget guard:** `run.py` aborts the optimize loop *before* any call once
cumulative spend reaches the ceiling (`--budget-usd`, default **$500**).

## Run

```bash
# gepa + deps live in the local (gitignored) scratch/gepa uv project
cd scratch/gepa
uv run --with litellm --with databricks-sdk --with networkx --with pandas \
    python <repo>/examples/otto/experiments/cloudcast/run.py \
    --run-dir <repo>/examples/otto/experiments/cloudcast/outputs/gemini3pro_full

# print config + pre-run cost projection only (no LLM calls):
... run.py --dry-run
```

## Result (single clean run, 2026-07-04)

Full artifacts in `outputs/gemini3pro_full/` (`results.json`, `best_program.py`,
`reflection_calls.json`, `candidates.json`, `candidate_tree.html`, `run.log`).

| Metric | Value |
|---|---|
| Seed (Dijkstra) mean cost across 5 configs | **$209.17** (mean score 0.0052) |
| Best evolved mean cost | **$154.31** (mean score 0.007592) |
| **Cost savings vs baseline** | **26.23%** |
| Objective | `score = 1/(1+cost)`, higher is better; ↑ score = ↓ cost |
| Proposer / reflection LM calls | **14** |
| `total_metric_calls` | 104 (GEPA finishes the in-flight minibatch past 100) |
| Candidates kept | 4 (`best_idx=3`) |
| Spend (USD) | **$0.6154** — all *estimated* (metered $0: no litellm price map) |
| Tokens in / out (billed, incl. reasoning) | 73,753 / 38,990 |
| Wallclock | 285.0 s (~4.75 min) |
| Budget-aborted? | No (ceiling $500, ~0.12% used) |

The evolved algorithm (`outputs/gemini3pro_full/best_program.py`) replaces the
seed's single Dijkstra path with a **Directed Steiner Tree heuristic
(Takahashi–Matsuyama)** that minimizes egress cost and adds a load-based penalty
to spread partitions across paths for throughput — the same algorithm family the
blog reports ("provider-aware Steiner tree").

**Fidelity note vs the blog's 40.2%:** same config (gemini-3-pro-preview,
`max_metric_calls=100`, 5-config valset) but GEPA's proposer is stochastic and
this single seed-0 run explored only 4 kept candidates in the budget, landing at
26.23%. The blog's 40.2% is its best reported trajectory; a real, honest single
run under the identical budget lands lower. Re-running (or raising the budget)
would explore more candidates.
