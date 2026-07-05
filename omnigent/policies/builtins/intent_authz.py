"""EXAMPLE: Intent-Based Authorization (IBA) via LLM-generated CEL scope.

This is a **pedagogical example policy**, not a hardened production control.
It demonstrates one idea end to end:

    1. On the FIRST user request of a session, an LLM reads the user's prompt
       and generates a small set of CEL boolean predicates describing which
       future tool calls fall *within the user's stated intent*.
    2. Those rules are cached in the engine's ``session_state``.
    3. On every SUBSEQUENT tool call to the Google Workspace MCP
       (``mcp__google__*`` / ``google__*``) or to a shell tool
       (``sys_os_shell`` / native ``Bash``), the cached CEL is replayed
       *deterministically* through the same engine that powers
       :mod:`omnigent.policies.builtins.cel`:

       - in-scope  → **ALLOW**
       - out-of-scope → **ASK** the user for permission (not a silent DENY).

The point of ASK (rather than DENY) is that the LLM classifies intent *once*,
up front, and the deterministic replay never blocks the user outright — it
escalates anything the up-front classification did not clearly authorize back
to a human.

Worked example
--------------
User's first prompt: *"make a Google Slides presentation about Lakebase"*.

The generator is expected to emit rules that allow the Google MCP tools (Drive
/ Docs / Sheets / Slides) plus read-only shell, e.g.::

    event.data.name.startsWith("mcp__google__")
        || event.data.name.startsWith("google__")
    (event.data.name == "sys_os_shell" || event.data.name == "Bash")
        && event.data.arguments.command.matches(
             "^\\s*(ls|cat|pwd|grep|find|head|tail|git status|git diff)\\b")

Later in the session:

    * ``mcp__google__slides_presentation_create`` → matches rule 1 → ALLOW.
    * ``sys_os_shell {command: "ls ."}``           → matches rule 2 → ALLOW.
    * ``sys_os_shell {command: "git push"}``       → matches nothing → ASK.
    * ``sys_os_shell {command: "gh pr create"}``   → matches nothing → ASK.

Design notes (why it looks the way it does)
-------------------------------------------
* **Reuse, don't reinvent.** The LLM only ever emits *boolean* CEL
  predicates. We combine them with ``||`` and wrap them in a tiny
  ``... ? {"result":"ALLOW"} : {"result":"ASK", ...}`` shell, then compile and
  evaluate that through :func:`omnigent.policies.builtins.cel.cel_policy` — the
  exact same CEL engine and event contract documented there
  (``event.type`` / ``event.target`` / ``event.data`` / ``event.context.*``).
* **Fail open / abstain.** With no server ``llm:`` config there is no
  ``event["llm_client"]``; generation is skipped and every tool call is
  allowed, mirroring :mod:`omnigent.policies.builtins.routing`.
* **Deterministic replay.** After the one-time generation, no further LLM
  calls happen — subsequent verdicts are pure CEL evaluation, and ALLOW
  verdicts are memoized per tool-call signature in ``session_state``.
* **Scope of this example.** Only Google MCP + shell tools are gated;
  every other tool abstains (ALLOW). Like the sibling
  :mod:`omnigent.policies.builtins.google` policy, this enforces at the
  tool-call boundary only and is *not* a complete sandbox.

YAML usage (zero required params)::

    policies:
      intent_authz:
        type: function
        handler: omnigent.policies.builtins.intent_authz.intent_based_authorization
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any, cast

from omnigent.policies.builtins.cel import cel_policy
from omnigent.policies.schema import PolicyCallable, PolicyEvent, PolicyResponse

_log = logging.getLogger(__name__)

# ── Session-state keys ────────────────────────────────────────────────────────
# All IBA state lives under a ``_iba_`` prefix so it is easy to spot in the
# conversation's persisted ``session_state``.

# Set to "1" once generation has run (even if it produced zero rules). Its
# presence is what flips the policy from "fail-open" to "enforcing".
_GENERATED_KEY = "_iba_generated"
# The verbatim first-request text, restated in ASK reasons.
_INTENT_KEY = "_iba_intent"
# The list[str] of LLM-generated CEL boolean predicates.
_RULES_KEY = "_iba_rules"
# Prefix for per-tool-call-signature ALLOW memoization: ``_iba_verdict:<sig>``.
_VERDICT_KEY_PREFIX = "_iba_verdict:"

# ── Which tools this example gates ────────────────────────────────────────────
# Google Workspace MCP tools (any server prefix) and the shell tools. Everything
# else abstains (ALLOW). Mirrors the tool-name matching in
# ``omnigent.policies.builtins.google`` / ``.working_dir``.
_GOOGLE_PREFIXES: tuple[str, ...] = ("mcp__google__", "google__")
_SHELL_TOOLS: frozenset[str] = frozenset({"sys_os_shell", "Bash"})

# Keep the intent short in the generator prompt / ASK reason / CEL literal.
_MAX_INTENT_CHARS = 400


# ── The LLM prompt (kept here, in full, so readers see exactly what is asked) ──
#
# We ask for *boolean* CEL predicates only — not full ``{"result": ...}`` maps.
# That keeps the model's job simple and keeps us in control of the ALLOW/ASK
# wrapping. The output shape is pinned by the structured-output schema below,
# so the prompt only has to describe the task and the available ``event`` fields.
_RULE_GEN_INSTRUCTIONS = """\
You are an authorization planner for an AI agent. You are given the user's
FIRST message of a session. Your job is to decide, up front, which future tool
calls clearly fall WITHIN the user's stated intent.

Emit a JSON list of CEL (Common Expression Language) boolean expressions. Each
expression is evaluated later against one tool call and must return TRUE when
that call is IN SCOPE for the user's intent. A tool call that matches NO
expression will be escalated to the user for approval (not silently blocked).

Only two families of tool calls are gated, so only write rules for these:
  * Google Workspace MCP tools — names start with "mcp__google__" or "google__"
    (Drive, Docs, Sheets, Slides, Gmail, Calendar).
  * Shell tools — event.data.name is "sys_os_shell" or "Bash", with the command
    string in event.data.arguments.command.

The CEL `event` variable has these fields:
  * event.type            -> "tool_call"
  * event.data.name       -> the tool name (string)
  * event.data.arguments  -> the tool arguments (map); for shell tools,
                             event.data.arguments.command is the command string.

Guidance:
  * Be generous with the Google MCP tools the intent needs (e.g. a "make a
    presentation" intent should allow Drive/Docs/Slides tools).
  * Allow ONLY read-only / inspection shell commands (ls, cat, pwd, grep, find,
    head, tail, echo, and read-only git like `git status` / `git diff` /
    `git log`). Do NOT write a rule that allows state-changing or outbound
    commands such as `git push`, `gh pr create`, `rm`, `curl`, or package
    installs — those should fall through to a user approval prompt.
  * Prefer a few broad rules over many narrow ones. Use CEL string helpers:
    startsWith(), contains(), matches() (RE2 regex).

Example rules for the intent "make a Google Slides presentation about Lakebase"
(each element is one CEL boolean expression):
  * Allow any Google Workspace MCP tool:
      event.data.name.startsWith("mcp__google__")
        || event.data.name.startsWith("google__")
  * Allow only read-only shell commands:
      (event.data.name == "sys_os_shell" || event.data.name == "Bash")
        && event.data.arguments.command.matches(
             "^\\\\s*(ls|cat|pwd|grep|find|head|tail|git status|git diff)\\\\b")
"""

# Structured-output schema: forces ``{"rules": ["<cel>", ...]}`` so there is no
# free-text parsing (same technique as routing.py / prompt.py).
_RULE_GEN_SCHEMA: dict[str, Any] = {  # type: ignore[explicit-any]  # opaque JSON-schema shape
    "format": {
        "type": "json_schema",
        "name": "iba_scope_rules",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "rules": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["rules"],
            "additionalProperties": False,
        },
    },
}


def _extract_response_text(response: Any) -> str:  # type: ignore[explicit-any]  # foreign LLM SDK response
    """Extract text from an LLM response (OpenAI SDK or omnigent shape).

    :param response: Response from ``PolicyLLMClient.create()``.
    :returns: The extracted text, or empty string when unrecognized/empty.
    """
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    output = getattr(response, "output", None)
    if not isinstance(output, list) or not output:
        return ""
    content = getattr(output[0], "content", None)
    if not isinstance(content, list) or not content:
        return ""
    return getattr(content[0], "text", "") or ""


# The engine injects a live ``llm_client`` object (and may inject other
# non-serializable objects) into *every* event dict — see
# ``omnigent.policies.function.py``. Those objects cannot be converted to CEL
# values, so handing the raw event to the CEL evaluator makes ``compiled.eval``
# return an ERROR-typed result and the CEL policy abstain (None). The replay
# only needs the documented CEL event contract, so project the event down to
# these keys before evaluation.
_CEL_EVENT_KEYS: tuple[str, ...] = ("type", "target", "data", "context", "session_state")


def _cel_safe_event(event: PolicyEvent) -> PolicyEvent:
    """Project *event* to the CEL-visible fields, dropping injected live objects.

    The engine stamps a live ``llm_client`` (and potentially other
    non-serializable objects) onto every event; cel-expr-python cannot convert
    those, which silently poisons evaluation. Returning an allowlisted copy
    keeps every field the generated CEL rules reference (``event.data.name`` and
    ``event.data.arguments.command``) while excluding the injected objects.

    :param event: The raw policy event as passed by the engine.
    :returns: A new dict containing only the CEL-safe keys that are present.
    """
    safe: dict[str, Any] = {  # type: ignore[explicit-any]  # projected event payload
        k: event[k]  # type: ignore[literal-required]
        for k in _CEL_EVENT_KEYS
        if k in event
    }
    return cast("PolicyEvent", safe)


def _is_gated_tool(name: str) -> bool:
    """Whether this example gates *name* (Google MCP or a shell tool).

    :param name: The raw tool name from ``event.data.name``.
    :returns: ``True`` for Google MCP / shell tools; ``False`` otherwise
        (the policy abstains on everything else).
    """
    return name in _SHELL_TOOLS or name.startswith(_GOOGLE_PREFIXES)


def _signature(name: str, arguments: Any) -> str:  # type: ignore[explicit-any]  # opaque tool-arg JSON
    """Stable short hash of a tool call, for per-signature ALLOW memoization.

    :param name: Tool name.
    :param arguments: Tool arguments (any JSON-able value).
    :returns: A 16-char hex digest.
    """
    try:
        payload = json.dumps({"n": name, "a": arguments}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = f"{name}:{arguments!r}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@lru_cache(maxsize=256)
def _scope_checker(rules_key: tuple[str, ...], intent: str) -> PolicyCallable:
    """Compile the combined in-scope CEL into a replayable evaluator.

    The generated boolean predicates are OR-ed together and wrapped so the
    result is a map the CEL engine understands: ALLOW when any rule matches,
    ASK (naming the tool and restating the intent) otherwise. Compilation and
    evaluation go through :func:`omnigent.policies.builtins.cel.cel_policy` —
    the same engine and event contract as the standalone CEL policy.

    Memoized (``lru_cache``) because the engine is rebuilt per evaluation, so a
    closure cannot hold the compiled expression across tool calls; keying on the
    (rules, intent) pair keeps compilation to once per distinct rule set.

    :param rules_key: The generated CEL predicates as a hashable tuple.
    :param intent: The original user intent, restated in the ASK reason.
    :returns: A synchronous policy callable returning an ALLOW/ASK response.
    :raises ValueError: If the combined expression fails to compile (the LLM
        emitted invalid CEL) — the caller catches this and falls back to ASK.
    """
    scope = " || ".join(f"({rule})" for rule in rules_key)
    # ``json.dumps`` yields a valid CEL string literal (CEL shares JSON's \" \\
    # \n \uXXXX escapes), so the intent can be safely inlined into the reason.
    intent_literal = json.dumps(intent)
    reason_expr = (
        '"Intent-Based Authorization: the tool " + event.data.name + '
        '" is outside the intent captured at the start of this session (intent: " + '
        f"{intent_literal}"
        ' + "). Approve to run this out-of-scope action?"'
    )
    expression = (
        f'({scope}) ? {{"result": "ALLOW"}} : {{"result": "ASK", "reason": {reason_expr}}}'
    )
    return cel_policy(expression=expression)


def intent_based_authorization() -> PolicyCallable:
    """Factory: build the Intent-Based Authorization example policy.

    Takes no parameters — attach it as-is. See the module docstring for the
    end-to-end flow.

    :returns: An async policy callable implementing the IBA example.
    """

    async def _evaluate(event: PolicyEvent) -> PolicyResponse | None:
        """Generate rules on the first request; replay them on tool calls.

        :param event: The policy event dict.
        :returns: A :class:`PolicyResponse`, or ``None`` to abstain (ALLOW).
        """
        phase = event.get("type")
        if phase == "request":
            return await _maybe_generate_rules(event)
        if phase == "tool_call":
            return _replay_on_tool_call(event)
        return None

    return _evaluate  # type: ignore[return-value]


async def _maybe_generate_rules(event: PolicyEvent) -> PolicyResponse | None:
    """On the first request only, ask the LLM for CEL scope rules and cache them.

    :param event: A ``request`` policy event (``data`` is the user message).
    :returns: ALLOW with ``state_updates`` caching intent + rules, or ``None``
        to abstain (already generated, empty message, or no LLM client).
    """
    state = event.get("session_state") or {}
    if state.get(_GENERATED_KEY):
        return None  # Only ever generate once per session.

    prompt = event.get("data")
    if not isinstance(prompt, str) or not prompt.strip():
        return None

    llm_client = event.get("llm_client")
    if llm_client is None:
        # No server llm: config — abstain (fail open), exactly like routing.py.
        _log.warning(
            "intent_based_authorization: event['llm_client'] is None — "
            "server has no llm: config. Abstaining (no intent scoping)."
        )
        return None

    intent = prompt.strip()[:_MAX_INTENT_CHARS]
    try:
        response = await llm_client.create(
            input=[
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ],
            instructions=_RULE_GEN_INSTRUCTIONS,
            text=_RULE_GEN_SCHEMA,
        )
        raw_text = _extract_response_text(response)
        parsed = json.loads(raw_text) if raw_text else {}
        rules = parsed.get("rules", []) if isinstance(parsed, dict) else []
    except Exception:  # noqa: BLE001 — catch-all for LLM/JSON failures; fail open.
        _log.exception("intent_based_authorization: rule generation failed; abstaining")
        return None

    rules = [r for r in rules if isinstance(r, str) and r.strip()]
    _log.info(
        "intent_based_authorization: generated %d scope rule(s) for intent %r",
        len(rules),
        intent,
    )
    return {
        "result": "ALLOW",
        "state_updates": [
            {"key": _GENERATED_KEY, "action": "set", "value": "1"},
            {"key": _INTENT_KEY, "action": "set", "value": intent},
            {"key": _RULES_KEY, "action": "set", "value": rules},
        ],
    }


def _replay_on_tool_call(event: PolicyEvent) -> PolicyResponse | None:
    """Deterministically replay the cached CEL against one tool call.

    :param event: A ``tool_call`` policy event.
    :returns: ALLOW (in scope), ASK (out of scope / unverifiable), or ``None``
        to abstain (non-gated tool, or rules never generated → fail open).
    """
    data = event.get("data")
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if not isinstance(name, str) or not _is_gated_tool(name):
        return None  # Out of this example's scope → ALLOW.

    state = event.get("session_state") or {}
    if not state.get(_GENERATED_KEY):
        # Intent scoping never ran (e.g. no LLM client) → fail open.
        return None

    intent = state.get(_INTENT_KEY, "") if isinstance(state.get(_INTENT_KEY), str) else ""
    rules = state.get(_RULES_KEY)
    rules = [r for r in rules if isinstance(r, str)] if isinstance(rules, list) else []

    # Per-signature ALLOW memoization: skip re-evaluating a call we already
    # cleared. (ASK verdicts are never cached — on ASK the engine withholds
    # state_updates until the user approves.)
    sig = _signature(name, data.get("arguments"))
    verdict_key = f"{_VERDICT_KEY_PREFIX}{sig}"
    if state.get(verdict_key) == "ALLOW":
        return {"result": "ALLOW"}

    if not rules:
        # Generation produced no in-scope rules → nothing is authorized → ASK.
        return _ask(name, intent)

    try:
        checker = _scope_checker(tuple(rules), intent)
        response = checker(_cel_safe_event(event))
    except Exception:  # noqa: BLE001 — bad CEL / eval error → fail safe to ASK.
        _log.exception("intent_based_authorization: scope check failed; asking user")
        return _ask(name, intent)

    if response is not None and response.get("result") == "ALLOW":
        return {
            "result": "ALLOW",
            "state_updates": [{"key": verdict_key, "action": "set", "value": "ALLOW"}],
        }

    # Out of scope (ASK) or an eval error that made CEL abstain (None) → ASK.
    if response is not None and response.get("result") == "ASK":
        return {"result": "ASK", "reason": response.get("reason") or _ask(name, intent)["reason"]}
    return _ask(name, intent)


def _ask(name: str, intent: str) -> PolicyResponse:
    """Build the ASK response naming the tool and restating the intent.

    :param name: The tool being gated, e.g. ``"sys_os_shell"``.
    :param intent: The original session intent.
    :returns: An ASK :class:`PolicyResponse`.
    """
    return {
        "result": "ASK",
        "reason": (
            f"Intent-Based Authorization: the tool {name!r} is outside the intent "
            f"captured at the start of this session (intent: {intent!r}). "
            f"Approve to run this out-of-scope action?"
        ),
    }


# ── Registry ─────────────────────────────────────────────────────────────────

POLICY_REGISTRY: list[dict[str, Any]] = [  # type: ignore[explicit-any]  # opaque registry-entry JSON
    {
        "handler": "omnigent.policies.builtins.intent_authz.intent_based_authorization",
        "kind": "factory",
        "name": "Intent-Based Authorization (LLM-generated CEL scope)",
        "description": (
            "EXAMPLE policy. On the first user request, an LLM generates CEL scope "
            "rules describing which future tool calls match the user's intent; the "
            "rules are cached in session_state and replayed deterministically on "
            "subsequent Google Workspace MCP (mcp__google__* / google__*) and shell "
            "(sys_os_shell / Bash) tool calls. In-scope calls ALLOW; out-of-scope "
            "calls ASK the user for approval. Requires the server to have an llm: "
            "config block; abstains (fail open) when absent."
        ),
        "params_schema": {
            "type": "object",
            "properties": {},
        },
    },
]
