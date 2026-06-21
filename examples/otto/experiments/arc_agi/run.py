#!/usr/bin/env python3
"""Run Otto's tiny GEPA ARC-AGI reproduction harness."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluator import (
    BACKGROUND,
    MODEL,
    OBJECTIVE,
    PROFILE,
    ensure_databricks_env,
    load_arc_slices,
    load_seed,
    make_evaluator,
)
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=4)
    parser.add_argument("--val-size", type=int, default=4)
    parser.add_argument("--max-metric-calls", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-llm-calls", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--run-dir", default=str(ROOT / "outputs" / "smoke"))
    parser.add_argument("--seed-agent", default=str(ROOT / "seed_agent.py"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and imports, then exit before LLM calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_candidate = load_seed(Path(args.seed_agent))
    train_set, val_set = load_arc_slices(args.train_size, args.val_size, args.seed)
    print(
        f"ARC slices: train={len(train_set)} val={len(val_set)} "
        f"budget={args.max_metric_calls} workers={args.max_workers} "
        f"model={args.model} profile={args.profile}"
    )

    if args.dry_run:
        print("Dry run complete: imports, seed artifact, and ARC slices loaded.")
        return

    ensure_databricks_env(args.profile)

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=args.run_dir,
            max_metric_calls=args.max_metric_calls,
            parallel=args.max_workers > 1,
            max_workers=args.max_workers,
            cache_evaluation=True,
            track_best_outputs=True,
        ),
        reflection=ReflectionConfig(reflection_lm=args.model),
    )

    result = optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=make_evaluator(
            model=args.model,
            max_llm_calls=args.max_llm_calls,
            profile=args.profile,
        ),
        dataset=train_set,
        valset=val_set,
        objective=OBJECTIVE,
        background=BACKGROUND,
        config=config,
    )

    output_dir = Path(args.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_agent.py"
    best_path.write_text(result.best_candidate, encoding="utf-8")
    print(f"Saved best candidate: {best_path}")
    if getattr(result, "val_aggregate_scores", None):
        baseline_score = result.val_aggregate_scores[0]
        best_score = result.val_aggregate_scores[result.best_idx]
        print(f"Baseline validation score: {baseline_score:.4f}")
        print(f"Best validation score: {best_score:.4f}")
    if getattr(result, "total_metric_calls", None) is not None:
        print(f"Total metric calls: {result.total_metric_calls}")


if __name__ == "__main__":
    main()
