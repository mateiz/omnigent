---
name: auto-research
description: Run an autoresearch-style baseline, edit, evaluate, and keep-or-revert loop over one editable artifact.
---

# auto-research — one artifact, one metric, fixed budget

Use this skill when the user wants an autonomous research loop like Karpathy's
`autoresearch`: repeatedly edit one artifact, evaluate it under a fixed budget,
and keep only metric-improving changes.

## Loop

1. Pick one editable artifact: a prompt, program, config, policy, or skill.
2. Pick one comparable metric and one fixed evaluation budget.
3. Run the baseline and record the score.
4. Propose one small edit with a clear hypothesis.
5. Evaluate the edited artifact with the same budget.
6. Keep the edit if the score improves; otherwise revert exactly that edit.
7. Append the decision to a log and continue until the budget is exhausted.

## Decision log

Keep a simple trail, for example:

```text
round=3
artifact=seed_agent.py
hypothesis=add shape-preserving fallback
baseline_score=0.25
candidate_score=0.50
decision=keep
notes=solved task 2 without regressing task 1
```

## Rules

- Change only the chosen artifact unless the user expands scope.
- Do not compare runs with different budgets, splits, or metrics.
- Prefer small edits that are easy to revert and explain.
- If evaluation fails, treat the score as worse and revert unless the failure is
  clearly caused by broken infrastructure.
- Stop when the fixed budget is spent; summarize the best artifact, best score,
  and strongest remaining hypothesis.
