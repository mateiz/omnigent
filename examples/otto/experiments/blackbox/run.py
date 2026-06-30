#!/usr/bin/env python3
"""Run the GEPA ``optimize_anything`` BLACKBOX example end-to-end.

This is the upstream ``gepa/examples/blackbox/main.py`` example (Single-Task
Search over the EvalSet benchmark) with ONE change: the reflection/proposer LM
is GPT-5 served through the **Databricks model gateway**, not api.openai.com.

Upstream config (unchanged):
    NUM_PROPOSALS      = 10
    EVALUATION_BUDGET  = 2000   (sub-second objective evals on a single problem)
    max_candidate_proposals = 20
    cache_evaluation   = True
    seed program       = minimal uniform-random single-sample solver
    proposer           = gpt-5  -> here: databricks/databricks-gpt-5

No LLM is in the objective-evaluation loop; the LM is only the GEPA proposer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import litellm
import numpy as np
from databricks.sdk.config import Config

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # make ``evalset`` and ``blackbox_lib`` importable

from blackbox_lib import (  # noqa: E402
    BACKGROUND,
    BudgetTracker,
    execute_code,
    extract_best_xs,
)
from evalset.problems import problem_configs, problems  # noqa: E402
from gepa.optimize_anything import (  # noqa: E402
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)

# --- Upstream example constants (gepa/examples/blackbox/main.py) ---
NUM_PROPOSALS = 10
EVALUATION_BUDGET = 2000
MAX_CANDIDATE_PROPOSALS = 20
SEED_CODE = """
import numpy as np

def solve(objective_function, config, best_xs=None):
    bounds = np.array(config['bounds'])
    all_attempts = []

    x = np.random.uniform(bounds[:, 0], bounds[:, 1])
    score = objective_function(x)
    all_attempts.append({"x": x.copy(), "score": score})

    return {"x": x, "score": score, "all_attempts": all_attempts}
"""

# --- Databricks gateway wiring ---
PROFILE = "oss"
DATABRICKS_MODEL = "databricks/databricks-gpt-5"  # GPT-5 served on the Databricks gateway
# GPT-5 public list pricing (USD / 1M tokens) used as a cost-estimate fallback
# when the gateway does not return metered cost metadata.
GPT5_INPUT_PER_MILLION = 1.25
GPT5_OUTPUT_PER_MILLION = 10.0


def databricks_credentials(profile: str) -> tuple[str, str]:
    """Resolve (serving-endpoints base url, bearer token) for the gateway.

    The ``oss`` profile authenticates via OAuth (``databricks-cli``), so there is
    no static PAT — we mint a short-lived bearer token from ``authenticate()``.
    LiteLLM's databricks provider expects ``api_base`` to point at
    ``{host}/serving-endpoints``.
    """
    cfg = Config(profile=profile)
    token = cfg.authenticate()["Authorization"].split(" ", 1)[1]
    api_base = cfg.host.rstrip("/") + "/serving-endpoints"
    return api_base, token


class DatabricksReflectionLM:
    """GPT-5 proposer/reflection LM routed through the Databricks gateway.

    Conforms to GEPA's ``LanguageModel`` protocol (callable(prompt) -> str) and
    exposes ``total_cost`` so GEPA uses our tracking rather than wrapping it.
    """

    def __init__(self, model: str = DATABRICKS_MODEL, profile: str = PROFILE) -> None:
        self.model = model
        self.api_base, self.token = databricks_credentials(profile)
        self.calls: list[dict] = []
        self.total_cost = 0.0
        self.total_tokens_in = 0
        self.total_tokens_out = 0

    def __call__(self, prompt, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        start = time.time()
        response = litellm.completion(
            model=self.model,
            custom_llm_provider="databricks",
            messages=messages,
            api_base=self.api_base,
            api_key=self.token,
            drop_params=True,
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        ptoks = (getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        ctoks = (getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        try:
            metered = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:  # noqa: BLE001 - cost metadata is best-effort.
            metered = 0.0
        estimated = (ptoks / 1e6 * GPT5_INPUT_PER_MILLION) + (ctoks / 1e6 * GPT5_OUTPUT_PER_MILLION)
        cost = max(metered, estimated)
        self.total_cost += cost
        self.total_tokens_in += ptoks
        self.total_tokens_out += ctoks
        self.calls.append(
            {
                "duration_s": time.time() - start,
                "prompt_tokens": ptoks,
                "completion_tokens": ctoks,
                "metered_cost": metered,
                "estimated_cost": estimated,
                "cost": cost,
            }
        )
        return content


def seed_baseline_objective(problem_index: int, seed: int = 0) -> dict:
    """Run the seed program once to establish the baseline objective value."""
    np.random.seed(seed)
    result = execute_code(
        code=SEED_CODE,
        problem_index=problem_index,
        budget=EVALUATION_BUDGET // NUM_PROPOSALS,
        best_xs=[],
        seed=seed,
    )
    # ``score`` is the negated objective (higher = better for GEPA).
    return {
        "gepa_score": result["score"],
        "objective": -result["score"] if result["success"] else None,
        "success": result["success"],
        "error": result.get("error", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-index", type=int, default=46,
                        help="EvalSet problem index (upstream default 46 = Sargan dim 5).")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--num-proposals", type=int, default=NUM_PROPOSALS)
    parser.add_argument("--evaluation-budget", type=int, default=EVALUATION_BUDGET)
    parser.add_argument("--max-candidate-proposals", type=int, default=MAX_CANDIDATE_PROPOSALS)
    args = parser.parse_args()

    pidx = args.problem_index
    pname = problem_configs[pidx]
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "outputs" / f"p{pidx}"
    run_dir.mkdir(parents=True, exist_ok=True)

    reflection_lm = DatabricksReflectionLM()

    print(f"[config] problem_index={pidx} ({pname}) dim={problems[pidx].dim}")
    print(f"[config] num_proposals={args.num_proposals} evaluation_budget={args.evaluation_budget} "
          f"max_candidate_proposals={args.max_candidate_proposals}")
    print(f"[config] reflection_lm={DATABRICKS_MODEL} via Databricks gateway host={reflection_lm.api_base}")

    baseline = seed_baseline_objective(pidx)
    print(f"[baseline] seed objective={baseline['objective']} (gepa_score={baseline['gepa_score']})")

    budget = BudgetTracker(args.evaluation_budget, args.num_proposals)

    def evaluate(candidate, opt_state):
        if budget.remaining <= 0:
            return -1e9, {"score": -1e9, "error": "No budget remaining"}
        candidate_budget = budget.per_candidate
        best_xs = extract_best_xs(opt_state)
        result = execute_code(
            code=candidate,
            problem_index=pidx,
            budget=candidate_budget,
            best_xs=best_xs,
        )
        budget.record(result)
        side_info = {
            "score": result["score"],
            "all_trials": result.get("all_trials", []),
            "stdout": result.get("stdout", ""),
            "error": result.get("error", ""),
            "traceback": result.get("traceback", ""),
            "budget_total": budget.total,
            "budget_used": budget.used,
            "proposal_total": args.num_proposals,
            "proposal_completed": budget.candidates,
        }
        return result["score"], side_info

    wall_start = time.time()
    result = optimize_anything(
        evaluator=evaluate,
        seed_candidate=SEED_CODE,
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=str(run_dir),
                max_candidate_proposals=args.max_candidate_proposals,
                track_best_outputs=True,
                cache_evaluation=True,
            ),
            reflection=ReflectionConfig(
                reflection_lm=reflection_lm,
            ),
        ),
        objective=(
            "Evolve Python code that minimizes a blackbox objective function "
            "using the available evaluation budget efficiently."
        ),
        background=BACKGROUND,
    )
    wallclock_s = time.time() - wall_start

    best_idx = result.best_idx
    best_gepa_score = result.val_aggregate_scores[best_idx]
    best_objective = -best_gepa_score
    best_program = result.best_candidate
    if isinstance(best_program, dict):
        best_program = best_program.get("code") or next(iter(best_program.values()))

    summary = {
        "problem_index": pidx,
        "problem": pname,
        "dim": problems[pidx].dim,
        "reflection_model": DATABRICKS_MODEL,
        "databricks_host": reflection_lm.api_base,
        "num_proposals": args.num_proposals,
        "evaluation_budget": args.evaluation_budget,
        "max_candidate_proposals": args.max_candidate_proposals,
        "baseline_seed_objective": baseline["objective"],
        "best_objective": best_objective,
        "improvement": (baseline["objective"] - best_objective) if baseline["objective"] is not None else None,
        "best_idx": best_idx,
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "objective_evals_used": budget.used,
        "proposals_completed": budget.candidates,
        "reflection_llm_calls": len(reflection_lm.calls),
        "reflection_total_cost_usd": reflection_lm.total_cost,
        "reflection_tokens_in": reflection_lm.total_tokens_in,
        "reflection_tokens_out": reflection_lm.total_tokens_out,
        "wallclock_s": wallclock_s,
        "val_aggregate_scores": list(result.val_aggregate_scores),
    }

    (run_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "best_program.py").write_text(best_program or "", encoding="utf-8")
    (run_dir / "reflection_calls.json").write_text(
        json.dumps(reflection_lm.calls, indent=2), encoding="utf-8"
    )

    print("\n=== RESULTS ===")
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
