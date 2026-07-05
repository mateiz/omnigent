"""Tests for :mod:`omnigent.policies.builtins.intent_authz`.

Covers the IBA example end to end:

- First ``request`` generates CEL rules (LLM stubbed) and caches them.
- A second ``request`` abstains (generation runs once).
- No ``llm_client`` abstains (fail open).
- An in-scope Google MCP call ALLOWs; an out-of-scope ``git push`` shell call
  returns ASK naming the tool and restating the intent.
- Non-gated tools and un-generated sessions abstain (fail open).
- The registry entry is well-formed.

Requires ``cel-expr-python`` for the replay path; those tests are skipped when
it is not installed.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from omnigent.policies.builtins.intent_authz import (
    _GENERATED_KEY,
    _INTENT_KEY,
    _RULES_KEY,
    POLICY_REGISTRY,
    intent_based_authorization,
)

from .helpers import tool_call_event

# Rules the stub generator "returns" and that a real session would cache.
_SLIDES_RULES = [
    'event.data.name.startsWith("mcp__google__") || event.data.name.startsWith("google__")',
    (
        '(event.data.name == "sys_os_shell" || event.data.name == "Bash") && '
        'event.data.arguments.command.matches("^\\\\s*(ls|cat|pwd|git status|git diff)\\\\b")'
    ),
]
_INTENT = "make a Google Slides presentation about Lakebase"


# ── Fakes (explicit, no MagicMock for the response) ──────────────────────────


class _FakeResponse:
    """Minimal stand-in exposing ``output_text`` (what the policy reads)."""

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class _FakeLLMClient:
    """Stub ``PolicyLLMClient`` returning a fixed response."""

    def __init__(self, response: _FakeResponse) -> None:
        self._mock_create = AsyncMock(return_value=response)

    async def create(self, **kwargs: Any) -> _FakeResponse:
        return await self._mock_create(**kwargs)


def _request_event(
    data: str,
    *,
    client: _FakeLLMClient | None,
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ``request`` event with an optional ``llm_client`` attached."""
    return {
        "type": "request",
        "target": None,
        "data": data,
        "context": {"actor": {}, "usage": {}},
        "session_state": session_state or {},
        "llm_client": client,
    }


def _generated_state(rules: list[str] | None = None) -> dict[str, Any]:
    """Session state as it looks AFTER first-request generation."""
    return {
        _GENERATED_KEY: "1",
        _INTENT_KEY: _INTENT,
        _RULES_KEY: _SLIDES_RULES if rules is None else rules,
    }


# ── Factory ──────────────────────────────────────────────────────────────────


def test_factory_returns_callable() -> None:
    """The factory produces a callable (no required params)."""
    assert callable(intent_based_authorization())


# ── First request → generate + cache ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_first_request_generates_and_caches_rules() -> None:
    """First request calls the LLM and caches intent + rules via state_updates.

    What breaks if this fails: no scope is ever generated, so the deterministic
    replay has nothing to enforce.
    """
    client = _FakeLLMClient(_FakeResponse(json.dumps({"rules": _SLIDES_RULES})))
    policy = intent_based_authorization()

    result = await policy(_request_event(_INTENT, client=client))

    assert result is not None
    assert result["result"] == "ALLOW"
    updates = {u["key"]: u["value"] for u in result["state_updates"]}
    assert updates[_GENERATED_KEY] == "1"
    assert updates[_INTENT_KEY] == _INTENT
    assert updates[_RULES_KEY] == _SLIDES_RULES
    # Structured output + instructions were forwarded to the LLM.
    call_kwargs = client._mock_create.call_args.kwargs
    assert "text" in call_kwargs and "instructions" in call_kwargs


@pytest.mark.asyncio
async def test_second_request_abstains() -> None:
    """Generation runs once — a later request with state present abstains.

    What breaks if this fails: every user turn re-runs the LLM generator.
    """
    client = _FakeLLMClient(_FakeResponse(json.dumps({"rules": _SLIDES_RULES})))
    policy = intent_based_authorization()

    result = await policy(
        _request_event("another message", client=client, session_state=_generated_state())
    )

    assert result is None
    client._mock_create.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_llm_client_abstains() -> None:
    """No server ``llm:`` config → no client → abstain (fail open).

    What breaks if this fails: the policy crashes instead of degrading when the
    server has no LLM configured.
    """
    policy = intent_based_authorization()
    result = await policy(_request_event(_INTENT, client=None))
    assert result is None


@pytest.mark.asyncio
async def test_empty_request_abstains() -> None:
    """An empty first message has nothing to scope → abstain."""
    client = _FakeLLMClient(_FakeResponse(json.dumps({"rules": []})))
    policy = intent_based_authorization()
    result = await policy(_request_event("   ", client=client))
    assert result is None
    client._mock_create.assert_not_awaited()


@pytest.mark.asyncio
async def test_generation_failure_abstains() -> None:
    """An LLM/JSON error during generation fails open (abstain)."""
    client = _FakeLLMClient(_FakeResponse("not json"))
    policy = intent_based_authorization()
    result = await policy(_request_event(_INTENT, client=client))
    assert result is None


# ── Deterministic replay on tool calls (needs cel-expr-python) ───────────────

pytest.importorskip("cel_expr_python", reason="cel-expr-python not installed")


@pytest.mark.asyncio
async def test_in_scope_google_call_allows() -> None:
    """An in-scope Google Slides MCP call is ALLOWed and memoized.

    What breaks if this fails: intent-consistent work gets blocked.
    """
    policy = intent_based_authorization()
    event = tool_call_event(
        "mcp__google__slides_presentation_create",
        {"title": "Lakebase"},
        session_state=_generated_state(),
    )
    result = await policy(event)

    assert result is not None
    assert result["result"] == "ALLOW"
    # ALLOW verdict is cached per signature.
    assert result["state_updates"][0]["value"] == "ALLOW"


@pytest.mark.asyncio
async def test_in_scope_readonly_shell_allows() -> None:
    """A read-only shell command permitted by the generated rule ALLOWs."""
    policy = intent_based_authorization()
    event = tool_call_event(
        "sys_os_shell", {"command": "git status"}, session_state=_generated_state()
    )
    result = await policy(event)
    assert result is not None
    assert result["result"] == "ALLOW"


@pytest.mark.asyncio
async def test_out_of_scope_git_push_asks() -> None:
    """An out-of-scope ``git push`` returns ASK naming the tool and intent.

    What breaks if this fails: actions outside the captured intent slip through
    without user approval.
    """
    policy = intent_based_authorization()
    event = tool_call_event(
        "sys_os_shell", {"command": "git push origin main"}, session_state=_generated_state()
    )
    result = await policy(event)

    assert result is not None
    assert result["result"] == "ASK"
    assert "sys_os_shell" in result["reason"]
    assert "Lakebase" in result["reason"]  # intent restated
    # ASK carries no state_updates (nothing cached until the user approves).
    assert "state_updates" not in result or not result["state_updates"]


class _LiveLLMClient:
    """Sentinel standing in for the live ``PolicyLLMClient`` the engine injects.

    The engine unconditionally stamps this object onto *every* event dict,
    including ``tool_call`` events. cel-expr-python cannot convert an opaque
    object like this to a CEL value, so if the raw event reaches the CEL
    evaluator, ``eval`` errors and the replay silently abstains. Being a plain
    object with no ``__dict__`` serialization hooks reproduces that faithfully.
    """


def _tool_call_with_live_llm_client(
    tool: str,
    arguments: dict[str, Any],
    session_state: dict[str, Any],
) -> dict[str, Any]:
    """A ``tool_call`` event in the *real* engine shape — with an ``llm_client``.

    The shared ``tool_call_event`` helper never attaches an ``llm_client``,
    which is exactly why the original bug slipped past the tests. This restores
    the production shape (see ``omnigent.policies.function`` — the engine sets
    ``event['llm_client']`` on every dispatched event).
    """
    event = dict(tool_call_event(tool, arguments, session_state=session_state))
    event["llm_client"] = _LiveLLMClient()
    return event


@pytest.mark.asyncio
async def test_in_scope_call_with_live_llm_client_allows() -> None:
    """REGRESSION: an in-scope call ALLOWs even with a live ``llm_client`` present.

    This reproduces the production event shape — the engine injects a live
    ``llm_client`` object into every ``tool_call`` event. Before the CEL-safe
    projection fix, that object poisoned ``cel_policy``'s ``eval`` (ERROR →
    abstain), so IBA saw ``None`` and treated the in-scope call as ASK, defeating
    the policy for every gated call. It must ALLOW.
    """
    policy = intent_based_authorization()
    event = _tool_call_with_live_llm_client(
        "mcp__google__slides_presentation_create",
        {"title": "Lakebase"},
        _generated_state(),
    )
    result = await policy(event)

    assert result is not None
    assert result["result"] == "ALLOW"


@pytest.mark.asyncio
async def test_out_of_scope_call_with_live_llm_client_asks() -> None:
    """An out-of-scope call still ASKs with a live ``llm_client`` present.

    Confirms the projection fix does not over-correct into blanket ALLOW: the
    genuine out-of-scope path (``git push``) must still escalate to the user.
    """
    policy = intent_based_authorization()
    event = _tool_call_with_live_llm_client(
        "sys_os_shell",
        {"command": "git push origin main"},
        _generated_state(),
    )
    result = await policy(event)

    assert result is not None
    assert result["result"] == "ASK"
    assert "sys_os_shell" in result["reason"]
    assert "Lakebase" in result["reason"]


@pytest.mark.asyncio
async def test_non_gated_tool_abstains() -> None:
    """A non-Google, non-shell tool is outside this example's scope → abstain."""
    policy = intent_based_authorization()
    event = tool_call_event("web_search", {"query": "lakebase"}, session_state=_generated_state())
    result = await policy(event)
    assert result is None


@pytest.mark.asyncio
async def test_ungenerated_session_fails_open() -> None:
    """A gated call before any generation (e.g. no LLM) fails open (abstain)."""
    policy = intent_based_authorization()
    event = tool_call_event("sys_os_shell", {"command": "git push"}, session_state={})
    result = await policy(event)
    assert result is None


@pytest.mark.asyncio
async def test_empty_rules_asks_all_gated() -> None:
    """Generation with zero rules gates every in-scope tool to ASK."""
    policy = intent_based_authorization()
    event = tool_call_event(
        "mcp__google__drive_file_get",
        {"file_id": "1AbC"},
        session_state=_generated_state(rules=[]),
    )
    result = await policy(event)
    assert result is not None
    assert result["result"] == "ASK"


# ── Registry ─────────────────────────────────────────────────────────────────


def test_registry_entry_well_formed() -> None:
    """The registry has one factory entry with the expected handler and no
    required params.

    What breaks if this fails: server startup won't discover the policy, or
    users would be forced to supply params it doesn't take.
    """
    assert len(POLICY_REGISTRY) == 1
    entry = POLICY_REGISTRY[0]
    assert entry["handler"] == (
        "omnigent.policies.builtins.intent_authz.intent_based_authorization"
    )
    assert entry["kind"] == "factory"
    assert entry["params_schema"].get("required", []) == []
