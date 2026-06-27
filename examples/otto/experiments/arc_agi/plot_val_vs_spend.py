#!/usr/bin/env python3
"""Plot validation pass@2 vs cumulative spend (USD) for two optimization runs:

  * autoresearch  (Gemini-proposed single-artifact edits)
  * GEPA #2       (faithful reference-seed run)

Both curves are reconstructed from the runs' own logs/ledgers and printed to
stdout so the (spend, val) pairs can be verified before plotting. The plotted
curve for each run is the monotonic best-so-far validation score vs the
cumulative TOTAL spend at the moment that score was measured.
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

GEPA_RUN = OUT / "gepa_refseed_20260623T163414Z"
GEPA_PHASE = GEPA_RUN / "phase2_faithful_blog_seed_20260624T192812Z"
AR_RUN = OUT / "autoresearch_parallel_20260622T235704Z"

PNG = GEPA_PHASE / "val_vs_spend.png"


def best_so_far(pairs):
    """Given (spend, val) in chronological order, return monotonic best-so-far."""
    out = []
    best = None
    for spend, val in pairs:
        if best is None or val > best:
            best = val
        out.append((spend, best))
    return out


# --------------------------------------------------------------------------
# autoresearch: decision_log.jsonl carries, per round, the candidate_score and
# the cumulative eval_spend_usd + proposal_spend_usd.
# --------------------------------------------------------------------------
def extract_autoresearch():
    rows = [json.loads(l) for l in open(AR_RUN / "decision_log.jsonl")]
    raw = []  # (cum_spend, candidate_val)
    for r in rows:
        spend = r["eval_spend_usd"] + r["proposal_spend_usd"]
        raw.append((spend, r["candidate_score"], r["round"], r["decision"]))
    # The seed/baseline starts the curve at its baseline score with ~0 spend.
    baseline = rows[0]["baseline_score"]
    print("\n=== autoresearch raw rounds (cum_spend, candidate_val, round, decision) ===")
    for spend, val, rnd, dec in raw:
        print(f"  round {rnd:2d}  spend=${spend:8.4f}  val={val:.4f}  [{dec}]")
    chrono = [(0.0, baseline)] + [(s, v) for s, v, _, _ in raw]
    return best_so_far(chrono), baseline


# --------------------------------------------------------------------------
# GEPA #2: run_log.txt logs per-task eval cost lines "[hash] ... cost=$X" and
# per-iteration "Valset score for new program: X" / "Base program full valset
# score: X". We accumulate candidate-eval cost as we walk the log, then map
# each cumulative candidate-cost level into cost_ledger.jsonl to read the true
# TOTAL cumulative_cost (which also includes interleaved reflection spend).
# --------------------------------------------------------------------------
COST_RE = re.compile(r"cost=\$([0-9.]+)")
BASE_RE = re.compile(r"Base program full valset score:\s*([0-9.]+)")
NEW_RE = re.compile(r"Valset score for new program:\s*([0-9.]+)")


def build_cand_to_total_map():
    """Return (cand_cum_levels, total_cum_levels) from the cost ledger in order,
    so we can map a cumulative candidate-eval spend to the true total spend."""
    rows = [json.loads(l) for l in open(GEPA_RUN / "cost_ledger.jsonl")]
    cand_cum = 0.0
    cand_levels = []
    total_levels = []
    for r in rows:
        if r["kind"] == "candidate_llm":
            cand_cum += r["cost"]
        cand_levels.append(cand_cum)
        total_levels.append(r["cumulative_cost"])
    return cand_levels, total_levels


def cand_to_total(cand_spend, cand_levels, total_levels):
    """First ledger point whose cumulative candidate cost >= cand_spend."""
    for c, t in zip(cand_levels, total_levels):
        if c >= cand_spend - 1e-9:
            return t
    return total_levels[-1]


def extract_gepa():
    cand_levels, total_levels = build_cand_to_total_map()
    cand_cum = 0.0
    chrono = []  # (total_spend, val)
    baseline = None
    with open(GEPA_RUN / "run_log.txt") as f:
        for line in f:
            m = COST_RE.search(line)
            if m:
                cand_cum += float(m.group(1))
                continue
            mb = BASE_RE.search(line)
            if mb:
                val = float(mb.group(1))
                if baseline is None:
                    baseline = val
                total = cand_to_total(cand_cum, cand_levels, total_levels)
                chrono.append((total, val, "base"))
                continue
            mn = NEW_RE.search(line)
            if mn:
                val = float(mn.group(1))
                total = cand_to_total(cand_cum, cand_levels, total_levels)
                chrono.append((total, val, "new"))
    print("\n=== GEPA #2 raw measurements (total_spend, val, kind) ===")
    for spend, val, kind in chrono:
        print(f"  spend=${spend:8.4f}  val={val:.4f}  [{kind}]")
    bsf = best_so_far([(s, v) for s, v, _ in chrono])
    return bsf, baseline


def main():
    ar_curve, ar_base = extract_autoresearch()
    gepa_curve, gepa_base = extract_gepa()

    print("\n=== autoresearch best-so-far curve (spend, val) ===")
    for s, v in ar_curve:
        print(f"  (${s:.4f}, {v:.4f})")
    print("\n=== GEPA #2 best-so-far curve (spend, val) ===")
    for s, v in gepa_curve:
        print(f"  (${s:.4f}, {v:.4f})")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        [s for s, _ in ar_curve], [v for _, v in ar_curve],
        marker="o", color="#1f77b4", label="autoresearch", drawstyle="steps-post",
    )
    ax.plot(
        [s for s, _ in gepa_curve], [v for _, v in gepa_curve],
        marker="s", color="#d62728", label="GEPA #2 (faithful seed)", drawstyle="steps-post",
    )

    # annotate final points
    ar_fs, ar_fv = ar_curve[-1]
    gp_fs, gp_fv = gepa_curve[-1]
    ax.annotate(f"{ar_fv:.3f}", (ar_fs, ar_fv), textcoords="offset points",
                xytext=(8, -14), color="#1f77b4", fontweight="bold")
    ax.annotate(f"{gp_fv:.3f}", (gp_fs, gp_fv), textcoords="offset points",
                xytext=(8, 6), color="#d62728", fontweight="bold")

    ax.set_xlabel("Cumulative spend (USD)")
    ax.set_ylabel("Validation pass@2")
    ax.set_title("ARC-AGI optimization: validation pass@2 vs cumulative spend")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PNG, dpi=150)
    print(f"\nSaved plot -> {PNG}")


if __name__ == "__main__":
    main()
