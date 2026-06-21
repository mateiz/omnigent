# Otto — Automatic Research Agent (Design)

## Purpose
Otto is a minimal example agent that performs **automatic research**: it iteratively improves a text artifact (a prompt, a program, an agent architecture) against a measurable objective, with no human in the loop between rounds. It exists to make two workflows trivial to run as examples:

1. **GEPA `optimize_anything`** — reflective optimization of any text artifact over execution traces.
2. **Autoresearch loop** (after Karpathy's `autoresearch`) — a baseline -> edit -> evaluate -> keep/revert loop over one editable file with a fixed eval budget.

## Non-goals
- Not an orchestrator: Otto has no sub-agents and opens no PRs. It runs local experiment loops.
- No heavy dependencies in the omnigent package: experiment deps (`gepa`, `datasets`, `litellm`) are optional and installed per-experiment.

## Shape
```
examples/otto/
  config.yaml                      # claude-sdk brain, os_env (shell+fs), 2 skills, no sub-agents
  DESIGN.md                        # this doc
  skills/
    optimize-anything/SKILL.md     # how to scaffold + run a GEPA optimize_anything experiment
    auto-research/SKILL.md         # the baseline->edit->eval->keep/revert loop
  experiments/
    arc_agi/                       # runnable ARC-AGI reproduction (optional deps)
      requirements.txt
      README.md
      seed_agent.py                # ~10-line seed program (the artifact)
      run.py                       # optimize_anything runner, budget-configurable
      evaluator.py                 # scores solved ARC tasks
```

## ARC-AGI reproduction
The blog reports Gemini 3 Flash improving **32.5% -> 89.5%** on the public ARC test set by using `optimize_anything` to discover an agent architecture, at `max_metric_calls=3000, max_workers=64`. Otto reproduces the *pipeline*: it evolves the seed agent program against ARC tasks served by `databricks-gemini-3-flash` (oss profile).

Budget is configurable. The **default is a tiny smoke config** (~4 train / 4 val tasks, `max_metric_calls~=10`, `max_workers~=2`) that runs cheaply end-to-end and demonstrates an improvement signal. The README documents scaling to the full 3000-call run that reaches ~89.5%.

## Model access
All LLM calls go through Databricks-served models on the **oss** profile (e.g. `databricks-gemini-3-flash`) via `litellm`. No provider keys are hard-coded.
