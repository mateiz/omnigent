#!/usr/bin/env python3
"""Run the GEPA ``optimize_anything`` CLOUDCAST example end-to-end.

This is the upstream ``gepa/examples/adrs/cloudcast/main.py`` example
(Generalization mode: broadcast-routing algorithm discovery for multi-cloud data
transfer) with ONE change: the reflection/proposer LM is **Gemini 3 Pro** served
through the **Databricks model gateway**, not the public Gemini API.

Blog-faithful config (see provenance in the printed banner and README):
    proposer / reflection_lm = gemini-3-pro-preview
        -> here: databricks/databricks-gemini-3-pro (Databricks gateway, oss profile)
    max_metric_calls         = 100
    dataset = valset         = 5 multi-cloud broadcast configs
                               (intra_aws, intra_azure, intra_gcp, inter_agz, inter_gaz2)
    reflection_minibatch_size = 3
    seed program             = baseline Dijkstra shortest-cost single-path router
    objective score          = 1 / (1 + total_cost)   (higher is better)

The objective evaluation is a pure Python cloud simulator (near-free); the LM is
only the GEPA proposer.  A hard USD budget guard aborts the optimize loop BEFORE
exceeding the ceiling (default $500).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import litellm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # make the vendored ``utils`` package importable

from gepa.optimize_anything import (  # noqa: E402
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    SideInfo,
    TrackingConfig,
    optimize_anything,
)

from utils.dataset import load_config_dataset  # noqa: E402
from utils.simulation import (  # noqa: E402
    FAILED_SCORE,
    evaluation_failure_info,
    evaluation_success_info,
    get_program_path,
    run_evaluation,
    syntax_failure_info,
    syntax_is_valid,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cloudcast")

# ---------------------------------------------------------------------------
# Databricks gateway wiring (same approach as the blackbox example)
# ---------------------------------------------------------------------------
PROFILE = "oss"
DATABRICKS_HOST = "https://ai-oss-ecosystem-integration-testing.cloud.databricks.com"
DATABRICKS_MODEL = "databricks/databricks-gemini-3-pro"  # gemini-3-pro-preview on the gateway

# Gemini 3 Pro list pricing (USD / 1M tokens), standard context tier (<=200K tokens).
# Used as the cost estimate; the gateway does NOT return metered cost for this
# endpoint (litellm has no price map for the underlying gemini-3.1-pro-preview),
# so this estimate is what governs the hard budget guard.
GEMINI3PRO_INPUT_PER_MILLION = 2.0
GEMINI3PRO_OUTPUT_PER_MILLION = 12.0

HARD_BUDGET_USD = 500.0  # experiment-wide hard ceiling; overridable via --budget-usd
_TOKEN_MAX_AGE_S = 1800  # refresh the OAuth bearer token if older than this


class BudgetExceeded(RuntimeError):
    """Raised to abort the optimize loop before spend exceeds the hard ceiling."""


def _mint_token(profile: str = PROFILE) -> str:
    """Mint a short-lived OAuth bearer token from the databricks CLI.

    The ``oss`` profile shares its host with another profile, which makes the
    SDK's hostname-based ``Config.authenticate()`` ambiguous; shelling out to the
    CLI with an explicit ``-p`` avoids that.
    """
    out = subprocess.run(
        ["databricks", "auth", "token", "-p", profile],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)["access_token"]


class DatabricksReflectionLM:
    """Gemini 3 Pro proposer/reflection LM routed through the Databricks gateway.

    Conforms to GEPA's ``LanguageModel`` protocol (callable(prompt) -> str) and
    exposes ``total_cost`` so GEPA uses our tracking rather than wrapping it.

    Cost accounting: output tokens are computed as ``total_tokens - prompt_tokens``
    (not ``completion_tokens``) so billed *reasoning* tokens are included — for
    this reasoning model ``total_tokens`` >> prompt+completion.  Per-call cost is
    ``max(metered, estimated)``; metered is best-effort (0 for this endpoint).
    A hard budget check runs before every call and aborts the loop.
    """

    def __init__(
        self,
        model: str = DATABRICKS_MODEL,
        profile: str = PROFILE,
        host: str = DATABRICKS_HOST,
        budget_usd: float = HARD_BUDGET_USD,
    ) -> None:
        self.model = model
        self.profile = profile
        self.api_base = host.rstrip("/") + "/serving-endpoints"
        self.budget_usd = budget_usd
        self._token = _mint_token(profile)
        self._token_minted_at = time.time()
        self.calls: list[dict] = []
        self.total_cost = 0.0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.aborted = False

    def _token_fresh(self) -> str:
        if time.time() - self._token_minted_at > _TOKEN_MAX_AGE_S:
            self._token = _mint_token(self.profile)
            self._token_minted_at = time.time()
        return self._token

    def __call__(self, prompt, **kwargs) -> str:
        # --- HARD BUDGET GUARD: abort BEFORE spending past the ceiling ---
        if self.total_cost >= self.budget_usd:
            self.aborted = True
            raise BudgetExceeded(
                f"Hard budget ${self.budget_usd:.2f} reached "
                f"(spent ${self.total_cost:.4f} over {len(self.calls)} calls); aborting."
            )

        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        start = time.time()
        response = litellm.completion(
            model=self.model,
            custom_llm_provider="databricks",
            messages=messages,
            api_base=self.api_base,
            api_key=self._token_fresh(),
            drop_params=True,
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        ptoks = (getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        ctoks = (getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        ttoks = (getattr(usage, "total_tokens", 0) or 0) if usage else 0
        # Billed output = everything that isn't prompt (includes reasoning tokens).
        out_billed = max(ctoks, ttoks - ptoks) if ttoks else ctoks
        try:
            metered = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:  # noqa: BLE001 - cost metadata is best-effort.
            metered = 0.0
        estimated = (
            ptoks / 1e6 * GEMINI3PRO_INPUT_PER_MILLION
            + out_billed / 1e6 * GEMINI3PRO_OUTPUT_PER_MILLION
        )
        cost = max(metered, estimated)
        self.total_cost += cost
        self.total_tokens_in += ptoks
        self.total_tokens_out += out_billed
        self.calls.append(
            {
                "duration_s": round(time.time() - start, 3),
                "prompt_tokens": ptoks,
                "completion_tokens": ctoks,
                "total_tokens": ttoks,
                "output_billed_tokens": out_billed,
                "metered_cost": metered,
                "estimated_cost": estimated,
                "cost": cost,
                "cumulative_cost": round(self.total_cost, 6),
            }
        )
        logger.info(
            "[proposer call %d] in=%d out_billed=%d cost=$%.4f cumulative=$%.4f",
            len(self.calls),
            ptoks,
            out_billed,
            cost,
            self.total_cost,
        )
        return content


# ---------------------------------------------------------------------------
# Artifact being evolved + objective / background (VERBATIM from upstream main.py)
# ---------------------------------------------------------------------------

INITIAL_PROGRAM = """import networkx as nx
import pandas as pd
import os
from typing import Dict, List


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, 'SingleDstPath'] = None):
        self.src = src
        self.dsts = dsts
        self.num_partitions = num_partitions
        if paths is not None:
            self.paths = paths
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)


def search_algorithm(src, dsts, G, num_partitions):
    \"\"\"
    Find broadcast paths from source to all destinations.

    Uses Dijkstra's shortest path algorithm based on cost as the edge weight.

    Args:
        src: Source node identifier (e.g., "aws:ap-northeast-1")
        dsts: List of destination node identifiers
        G: NetworkX DiGraph with cost and throughput edge attributes
        num_partitions: Number of data partitions

    Returns:
        BroadCastTopology object with paths for all destinations and partitions
    \"\"\"
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        path = nx.dijkstra_path(h, src, dst, weight="cost")
        for i in range(0, len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])

    return bc_topology
"""

OPTIMIZATION_OBJECTIVE = """Optimize a broadcast routing algorithm for multi-cloud data transfer.

The algorithm decides how to route data from a single source to multiple destinations
across cloud providers (AWS, GCP, Azure). The goal is to minimize total cost
(egress fees + instance costs) while maintaining good transfer times."""

OPTIMIZATION_BACKGROUND = """Key information about the problem domain:

- The network is represented as a directed graph where:
  - Nodes are cloud regions (e.g., "aws:us-east-1", "gcp:europe-west1-a", "azure:eastus")
  - Edges have 'cost' ($/GB for egress) and 'throughput' (Gbps bandwidth) attributes

- Data is partitioned into num_partitions chunks that can be routed independently
- Each partition can take a different path to reach each destination
- Total cost = egress costs (data_vol × edge_cost) + instance costs (runtime × cost_per_hour)

- The algorithm must return a BroadCastTopology object containing:
  - paths[dst][partition] = list of edges [[src, dst, edge_data], ...]
  - Each destination must have at least one valid path for each partition

Evaluation feedback format:
- Cost: Total transfer cost in dollars
- Transfer time: Maximum time for all destinations to receive data (seconds)

Optimization targets:
1. Reduce total cost (egress + instance costs)
2. Find paths that balance cost and throughput
3. Consider multipath routing for better bandwidth utilization
4. Exploit cloud provider pricing differences (e.g., intra-provider is cheaper)"""

DATASET_ROOT = ROOT / "utils" / "cloudcast" / "config"


# ---------------------------------------------------------------------------
# Evaluator — called by GEPA for every (candidate, example) pair (VERBATIM logic)
# ---------------------------------------------------------------------------

def evaluate(candidate: dict, example: dict, **kwargs) -> tuple[float, SideInfo]:
    program_path = get_program_path(candidate["program"])

    if not syntax_is_valid(program_path):
        return FAILED_SCORE, syntax_failure_info(example)

    success, cost, transfer_time, error, details = run_evaluation(
        program_path, example["config_file"], example["num_vms"]
    )

    if not success:
        return FAILED_SCORE, evaluation_failure_info(error, example)

    score = 1.0 / (1.0 + cost)
    return score, evaluation_success_info(score, cost, transfer_time, example, details)


def _raw_cost(candidate: dict, example: dict) -> tuple[float, float]:
    """Return (score, raw_cost) for one config; raw_cost = inf on failure."""
    program_path = get_program_path(candidate["program"])
    if not syntax_is_valid(program_path):
        return FAILED_SCORE, float("inf")
    success, cost, _t, _err, _d = run_evaluation(
        program_path, example["config_file"], example["num_vms"]
    )
    if not success:
        return FAILED_SCORE, float("inf")
    return 1.0 / (1.0 + cost), cost


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    dataset = load_config_dataset(config_dir=args.config_dir)
    if not dataset:
        logger.error(f"No configuration files found in: {args.config_dir}")
        sys.exit(1)

    reflection_lm = DatabricksReflectionLM(budget_usd=args.budget_usd)

    # --- Seed baseline (pure simulator, no LLM) ---
    seed_candidate = {"program": INITIAL_PROGRAM}
    seed_scores, seed_costs = [], []
    for ex in dataset:
        s, c = _raw_cost(seed_candidate, ex)
        seed_scores.append(s)
        seed_costs.append(c)
    seed_mean_score = sum(seed_scores) / len(seed_scores)
    seed_mean_cost = sum(seed_costs) / len(seed_costs)

    # --- Config banner + pre-run cost projection ---
    banner = {
        "example": "CloudCast (GEPA optimize_anything, Generalization mode)",
        "provenance": {
            "config": "scratch/gepa/examples/adrs/cloudcast/main.py",
            "blog_snippet": ("scratch/gepa/docs/docs/blog/posts/2026-02-18-"
                             "introducing-optimize-anything/appendix/appendix-d-cloudcast.snippet"),
        },
        "reflection_model_blog": "gemini-3-pro-preview",
        "reflection_model_gateway": DATABRICKS_MODEL,
        "databricks_host": DATABRICKS_HOST,
        "max_metric_calls": args.max_metric_calls,
        "reflection_minibatch_size": args.minibatch_size,
        "skip_perfect_score": False,
        "seed": 0,
        "num_configs": len(dataset),
        "configs": [Path(ex["config_file"]).stem for ex in dataset],
        "objective_score": "1 / (1 + total_cost)  (higher is better)",
        "seed_baseline_mean_score": round(seed_mean_score, 6),
        "seed_baseline_mean_cost_usd": round(seed_mean_cost, 4),
        "hard_budget_usd": args.budget_usd,
    }
    # Projection: worst-case number of proposer calls is bounded by max_metric_calls;
    # in practice each proposal costs a handful of metric calls, so calls << budget.
    est_cost_per_call = 0.30  # generous upper estimate for a large Gemini-3-Pro reflection turn
    proj_calls_hi = args.max_metric_calls  # hard upper bound (1 proposer call cannot cost < 1 metric call)
    projection = {
        "assumed_cost_per_proposer_call_usd": est_cost_per_call,
        "worst_case_proposer_calls_upper_bound": proj_calls_hi,
        "worst_case_projected_spend_usd": round(est_cost_per_call * proj_calls_hi, 2),
        "note": ("Upper bound assumes every metric call were an LLM call, which it is not "
                 "(objective evals are pure simulator). Real proposer calls are typically "
                 "10-25, projected spend ~$2-8."),
    }
    print("\n=== CLOUDCAST CONFIG (blog-faithful) ===")
    print(json.dumps(banner, indent=2))
    print("\n=== PRE-RUN COST PROJECTION ===")
    print(json.dumps(projection, indent=2))

    if est_cost_per_call * proj_calls_hi > args.budget_usd:
        logger.error("Projected worst-case spend exceeds hard budget; refusing to launch.")
        sys.exit(2)

    if args.dry_run:
        print("\n[dry-run] Config + projection printed; endpoint verified separately. Exiting.")
        return

    run_dir = _resolve_run_dir(args.run_dir)
    (run_dir / "config.json").write_text(json.dumps({**banner, "projection": projection}, indent=2))
    logger.info(f"Run directory: {run_dir}")

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(run_dir),
            seed=0,
            max_metric_calls=args.max_metric_calls,
            track_best_outputs=True,
            use_cloudpickle=True,
            display_progress_bar=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=args.minibatch_size,
            reflection_lm=reflection_lm,
            skip_perfect_score=False,
        ),
        tracking=TrackingConfig(use_wandb=False),
    )

    logger.info("Starting GEPA optimization for CloudCast")
    wall_start = time.time()
    aborted = False
    result = None
    try:
        result = optimize_anything(
            seed_candidate=seed_candidate,
            evaluator=evaluate,
            dataset=dataset,
            valset=dataset,
            objective=OPTIMIZATION_OBJECTIVE,
            background=OPTIMIZATION_BACKGROUND,
            config=gepa_config,
        )
    except BudgetExceeded as e:
        aborted = True
        logger.error("BUDGET ABORT: %s", e)
    wallclock_s = time.time() - wall_start

    _write_results(run_dir, result, reflection_lm, dataset, seed_candidate,
                   seed_mean_score, seed_mean_cost, banner, wallclock_s, aborted, args)


def _write_results(run_dir, result, reflection_lm, dataset, seed_candidate,
                   seed_mean_score, seed_mean_cost, banner, wallclock_s, aborted, args) -> None:
    best_program_code = INITIAL_PROGRAM
    best_mean_score = seed_mean_score
    best_mean_cost = seed_mean_cost
    best_idx = None
    num_candidates = 0
    total_metric_calls = None

    if result is not None:
        best = result.best_candidate
        best_program_code = best["program"] if isinstance(best, dict) else best
        best_idx = getattr(result, "best_idx", None)
        num_candidates = getattr(result, "num_candidates", len(getattr(result, "candidates", [])))
        total_metric_calls = getattr(result, "total_metric_calls", None)
        best_cand = {"program": best_program_code}
        bs, bc = [], []
        for ex in dataset:
            s, c = _raw_cost(best_cand, ex)
            bs.append(s)
            bc.append(c)
        best_mean_score = sum(bs) / len(bs)
        best_mean_cost = sum(bc) / len(bc)

    cost_savings_pct = (
        (seed_mean_cost - best_mean_cost) / seed_mean_cost * 100.0 if seed_mean_cost else 0.0
    )

    summary = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "aborted_by_budget_guard": aborted,
        "reflection_model_blog": "gemini-3-pro-preview",
        "reflection_model_gateway": DATABRICKS_MODEL,
        "databricks_host": DATABRICKS_HOST,
        "max_metric_calls": args.max_metric_calls,
        "reflection_minibatch_size": args.minibatch_size,
        "num_configs": len(dataset),
        # --- objective: score = 1/(1+cost), higher is better ---
        "seed_baseline_mean_score": round(seed_mean_score, 6),
        "best_mean_score": round(best_mean_score, 6),
        "seed_baseline_mean_cost_usd": round(seed_mean_cost, 4),
        "best_mean_cost_usd": round(best_mean_cost, 4),
        "cost_savings_pct_vs_baseline": round(cost_savings_pct, 2),
        # --- run bookkeeping ---
        "best_idx": best_idx,
        "num_candidates": num_candidates,
        "total_metric_calls": total_metric_calls,
        # --- spend ---
        "proposer_llm_calls": len(reflection_lm.calls),
        "spend_total_usd": round(reflection_lm.total_cost, 6),
        "spend_metered_usd": round(sum(c["metered_cost"] for c in reflection_lm.calls), 6),
        "spend_estimated_usd": round(sum(c["estimated_cost"] for c in reflection_lm.calls), 6),
        "tokens_in": reflection_lm.total_tokens_in,
        "tokens_out_billed": reflection_lm.total_tokens_out,
        "hard_budget_usd": args.budget_usd,
        "wallclock_s": round(wallclock_s, 1),
    }

    (run_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "best_program.py").write_text(best_program_code or "", encoding="utf-8")
    (run_dir / "reflection_calls.json").write_text(
        json.dumps(reflection_lm.calls, indent=2), encoding="utf-8"
    )

    print("\n=== RESULTS ===")
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to: {run_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CloudCast broadcast optimization via Databricks Gemini 3 Pro.")
    p.add_argument("--max-metric-calls", type=int, default=100, help="Max fitness evaluations (blog: 100).")
    p.add_argument("--minibatch-size", type=int, default=3, help="Reflection minibatch size (upstream: 3).")
    p.add_argument("--budget-usd", type=float, default=HARD_BUDGET_USD, help="Hard spend ceiling.")
    p.add_argument("--config-dir", type=Path, default=DATASET_ROOT, help="Config files directory.")
    p.add_argument("--run-dir", type=Path, default=None, help="Output directory for artifacts.")
    p.add_argument("--dry-run", action="store_true", help="Print config + projection and exit.")
    return p.parse_args()


def _resolve_run_dir(run_dir_arg: Path | None) -> Path:
    if run_dir_arg is not None:
        run_dir = run_dir_arg
    else:
        run_dir = ROOT / "outputs" / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


if __name__ == "__main__":
    main()
