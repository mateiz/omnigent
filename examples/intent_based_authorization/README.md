# Intent-Based Authorization (IBA) — example policy

A small, pedagogical Omnigent policy that authorizes tool calls against the
**intent the user stated at the start of the session** — not a static allowlist
written ahead of time.

The policy lives in
[`omnigent/policies/builtins/intent_authz.py`](../../omnigent/policies/builtins/intent_authz.py);
this directory holds a runnable example config and this walkthrough.

## The idea

1. **First request → generate scope (once, with an LLM).**
   On the first user message of a session, the policy hands that message to the
   server-level LLM and asks it to emit a few **CEL boolean predicates**
   describing which future tool calls are *within the user's intent*. The intent
   text and the generated rules are cached in the policy's `session_state`.

2. **Every later tool call → replay the scope (deterministically, no LLM).**
   On each subsequent call to a **Google Workspace MCP** tool
   (`mcp__google__*` / `google__*`) or a **shell** tool (`sys_os_shell` /
   native `Bash`), the cached CEL is evaluated through the **same engine** used
   by the built-in [`cel_policy`](../../omnigent/policies/builtins/cel.py):

   | Result | Meaning |
   | --- | --- |
   | any rule matches | **ALLOW** — in scope |
   | no rule matches | **ASK** — escalate to the user for approval |

   ASK (not DENY) is deliberate: the up-front classification happens once, and
   anything it didn't clearly authorize is surfaced to a human rather than
   silently blocked. Every other tool (non-Google, non-shell) simply ALLOWs.

```
  first "request" event                subsequent "tool_call" events
  ─────────────────────                ─────────────────────────────
  user prompt ──► LLM ──► CEL rules ──► session_state ──► CEL replay ──► ALLOW / ASK
                          (cached)                        (no LLM)
```

## Worked example

First prompt: **"make a Google Slides presentation about Lakebase"**. The
generator is expected to produce rules like:

```cel
event.data.name.startsWith("mcp__google__") || event.data.name.startsWith("google__")
(event.data.name == "sys_os_shell" || event.data.name == "Bash")
    && event.data.arguments.command.matches("^\\s*(ls|cat|pwd|grep|find|head|tail|git status|git diff|git log)\\b")
```

Then, during the session:

| Tool call | Verdict |
| --- | --- |
| `mcp__google__slides_presentation_create` | ALLOW (rule 1) |
| `sys_os_shell {command: "ls ."}` | ALLOW (rule 2) |
| `sys_os_shell {command: "git push"}` | **ASK** — outside "make a presentation" |
| `sys_os_shell {command: "gh pr create"}` | **ASK** — outside "make a presentation" |

The ASK reason names the tool and restates the captured intent, so the user has
the context to approve (or reject) the out-of-scope action.

## Running it

The rule generator uses the **server-level LLM client**, so the server must be
started with an `llm:` config block. (Without one, the policy abstains — fail
open — and nothing is gated.)

1. Start the server with an `llm:` block, e.g.:

   ```yaml
   # server_config.yaml
   llm:
     model: openai/gpt-4o-mini
     # ...connection/credentials for your provider...
   ```

   ```bash
   omnigent server --config server_config.yaml
   ```

2. Run the example agent (point its `google` MCP entry at your real Google
   Workspace MCP server first):

   ```bash
   python -m omnigent examples/intent_based_authorization/agent.yaml \
       --prompt "Make a Google Slides presentation about Lakebase, then run
                 \`git status\` and push the result with \`git push\`."
   ```

   The Google Slides/Docs/Drive calls and `git status` run; `git push` is
   escalated to you for approval.

## Notes & limits

- **Example, not a sandbox.** Like the sibling
  [`google`](../../omnigent/policies/builtins/google.py) policy, this enforces
  at the tool-call boundary only. It does not cover other paths to the same data
  (e.g. shell hitting the Google REST API directly). Favor clarity over
  completeness — this is a teaching example.
- **Deterministic after generation.** Exactly one LLM call per session (the
  generation); every verdict after that is pure CEL. ALLOW verdicts are
  memoized per tool-call signature in `session_state`.
- **Fail open.** No `llm:` config → no `event["llm_client"]` → generation is
  skipped and all calls are allowed, mirroring
  [`routing`](../../omnigent/policies/builtins/routing.py).
