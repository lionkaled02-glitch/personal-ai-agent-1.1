"""Permission levels and the permission manager.

Design goals (see SECURITY.md):

- Fail-safe: anything that requires approval but has no approval channel is
  DENIED, never silently allowed.
- Policy-driven: defaults are LOW=allowed, MEDIUM/HIGH=approval required.
  The policy is plain data, so stricter deployments can deny levels outright.
- No shell or arbitrary-action tool is allowed. Explicit Phase 6 computer
  interactions are MEDIUM; destructive or externally consequential computer
  actions are HIGH and must use this same approval mechanism.

This module must not import from ``tools`` (tools depend on this module for
``PermissionLevel``), so :meth:`PermissionManager.check` accepts anything
matching the structural :class:`ToolDescriptor` protocol.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Protocol

from pydantic import BaseModel, Field


class PermissionLevel(IntEnum):
    """Ordering matters: higher level => stricter handling."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class PermissionDecision(StrEnum):
    ALLOWED = "ALLOWED"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"
    DENIED = "DENIED"


class ToolDescriptor(Protocol):
    """Structural view of a tool that the permission manager needs."""

    name: str
    permission_level: PermissionLevel


class ApprovalRequest(BaseModel):
    """Everything a human (or policy engine) needs to approve a step."""

    task_id: str
    step_id: str
    tool_name: str
    permission_level: PermissionLevel
    reason: str


class ApprovalCallback(Protocol):
    """Returns True to allow, False to deny a permissioned step."""

    def __call__(self, request: ApprovalRequest) -> bool: ...


class PermissionPolicy(BaseModel):
    """Per-level policy. The explicit deny-list always wins."""

    low: PermissionDecision = PermissionDecision.ALLOWED
    medium: PermissionDecision = PermissionDecision.REQUIRES_APPROVAL
    high: PermissionDecision = PermissionDecision.REQUIRES_APPROVAL
    denied_tools: frozenset[str] = Field(default_factory=frozenset)

    def decision_for(self, level: PermissionLevel) -> PermissionDecision:
        return {
            PermissionLevel.LOW: self.low,
            PermissionLevel.MEDIUM: self.medium,
            PermissionLevel.HIGH: self.high,
        }[level]


class PermissionManager:
    """Checks tool permission decisions and routes approval requests."""

    def __init__(
        self,
        policy: PermissionPolicy | None = None,
        approval: ApprovalCallback | None = None,
    ) -> None:
        self._policy = policy or PermissionPolicy()
        self._approval = approval

    @property
    def policy(self) -> PermissionPolicy:
        return self._policy

    def check(self, tool: ToolDescriptor) -> PermissionDecision:
        if tool.name in self._policy.denied_tools:
            return PermissionDecision.DENIED
        return self._policy.decision_for(tool.permission_level)

    def request_approval(self, request: ApprovalRequest) -> bool:
        """Ask the approval channel. Fail-safe: no channel => denied."""
        if self._approval is None:
            return False
        return bool(self._approval(request))


@dataclass(frozen=True)
class _PermissionAuthorization:
    """An executor-checked permission scope for one tool invocation.

    Computer tools also enforce permissions when called outside the agent
    executor (for example, through a direct registry call). When invoked by
    the executor, this short-lived scope lets the computer runtime reuse the
    already completed check/approval without prompting twice. A scope only
    covers the exact tool name and permission level; a MEDIUM approval can
    never authorize a HIGH operation.
    """

    task_id: str
    step_id: str
    tool_name: str
    permission_level: PermissionLevel


_current_permission_authorization: ContextVar[_PermissionAuthorization | None] = ContextVar(
    "agent_core_permission_authorization", default=None
)


def _active_permission_authorization() -> _PermissionAuthorization | None:
    """Return the currently active, executor-checked tool authorization."""
    return _current_permission_authorization.get()


@contextmanager
def _permission_authorization_scope(
    *,
    task_id: str,
    step_id: str,
    tool_name: str,
    permission_level: PermissionLevel,
) -> Iterator[None]:
    """Install a narrowly scoped authorization after the executor approves.

    This is deliberately private: callers should use ``PermissionManager``
    and the executor, not manufacture an authorization context themselves.
    The context is reset even when a tool raises.
    """
    authorization = _PermissionAuthorization(
        task_id=task_id,
        step_id=step_id,
        tool_name=tool_name,
        permission_level=permission_level,
    )
    token: Token[_PermissionAuthorization | None] = _current_permission_authorization.set(
        authorization
    )
    try:
        yield
    finally:
        _current_permission_authorization.reset(token)
