"""Exception hierarchy for the agent core.

All agent failures derive from :class:`AgentCoreError` so callers can catch
agent-level problems without accidentally swallowing unrelated errors.
"""

from __future__ import annotations


class AgentCoreError(Exception):
    """Base class for all agent core errors."""


class TaskStateError(AgentCoreError):
    """An illegal task or step state transition was attempted."""


class PlanningError(AgentCoreError):
    """The planner produced an invalid or unusable plan (or failed)."""


class ToolNotFoundError(AgentCoreError):
    """A tool was requested that is not registered."""


class ToolAlreadyRegisteredError(AgentCoreError, ValueError):
    """A tool name was registered twice.

    Also subclasses :class:`ValueError` for backwards compatibility with
    Phase 0 callers that catch the original exception type.
    """


class ToolInputError(AgentCoreError):
    """Tool input did not match the tool's declared input schema."""


class ToolExecutionError(AgentCoreError):
    """A registry-level tool execution failure.

    Faulty tool *results* normally surface as ``ToolResult(ok=False)``; this
    is reserved for unrecoverable failures at the registry boundary.
    """


class PermissionDeniedError(AgentCoreError):
    """Execution was attempted without an ALLOWED permission decision.

    Raised by the :class:`~agent_core.tool_runtime.ToolRuntime` backstop:
    no tool can execute through the runtime unless the caller passes an
    explicit :data:`~agent_core.permissions.PermissionDecision.ALLOWED`.
    """


class ProviderError(AgentCoreError):
    """A model provider could not service a request.

    Base for all provider-level failures. Non-transient by default: the
    gateway does not retry plain :class:`ProviderError`.
    """


class TransientProviderError(ProviderError):
    """A provider failure that is safe to retry (timeout, 5xx, rate limit,
    connection error). Retried by the model gateway with backoff."""


class ProviderConfigurationError(ProviderError):
    """Provider configuration is missing or invalid (unknown provider,
    missing credentials, vendor SDK not installed). Not retryable."""
