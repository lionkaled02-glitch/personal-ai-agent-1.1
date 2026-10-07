"""Permission system tests: levels, policy, fail-safe approval behavior."""

from __future__ import annotations

from agent_core import (
    ApprovalRequest,
    PermissionDecision,
    PermissionLevel,
    PermissionManager,
    PermissionPolicy,
    ToolSpec,
)


def make_spec(name: str, level: PermissionLevel) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} tool",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        permission_level=level,
    )


class TestDefaultPolicy:
    def test_low_allowed(self) -> None:
        manager = PermissionManager()
        assert manager.check(make_spec("t", PermissionLevel.LOW)) is PermissionDecision.ALLOWED

    def test_medium_requires_approval(self) -> None:
        manager = PermissionManager()
        assert (
            manager.check(make_spec("t", PermissionLevel.MEDIUM))
            is PermissionDecision.REQUIRES_APPROVAL
        )

    def test_high_requires_approval(self) -> None:
        manager = PermissionManager()
        assert (
            manager.check(make_spec("t", PermissionLevel.HIGH))
            is PermissionDecision.REQUIRES_APPROVAL
        )


class TestCustomPolicy:
    def test_high_can_be_denied_outright(self) -> None:
        policy = PermissionPolicy(high=PermissionDecision.DENIED)
        manager = PermissionManager(policy=policy)
        assert manager.check(make_spec("t", PermissionLevel.HIGH)) is PermissionDecision.DENIED

    def test_deny_list_wins_over_level_policy(self) -> None:
        policy = PermissionPolicy(denied_tools=frozenset({"nasty_tool"}))
        manager = PermissionManager(policy=policy)
        assert (
            manager.check(make_spec("nasty_tool", PermissionLevel.LOW)) is PermissionDecision.DENIED
        )
        assert (
            manager.check(make_spec("harmless", PermissionLevel.LOW)) is PermissionDecision.ALLOWED
        )


class TestApproval:
    def test_approval_granted(self) -> None:
        seen: list[ApprovalRequest] = []

        def approver(request: ApprovalRequest) -> bool:
            seen.append(request)
            return True

        manager = PermissionManager(approval=approver)
        request = ApprovalRequest(
            task_id="t",
            step_id="s",
            tool_name="medium_tool",
            permission_level=PermissionLevel.MEDIUM,
            reason="needs it",
        )
        assert manager.request_approval(request) is True
        assert seen == [request]

    def test_approval_denied(self) -> None:
        manager = PermissionManager(approval=lambda _request: False)
        request = ApprovalRequest(
            task_id="t",
            step_id="s",
            tool_name="medium_tool",
            permission_level=PermissionLevel.MEDIUM,
            reason="needs it",
        )
        assert manager.request_approval(request) is False

    def test_no_approval_channel_is_fail_safe_denied(self) -> None:
        manager = PermissionManager()  # no approval callback
        request = ApprovalRequest(
            task_id="t",
            step_id="s",
            tool_name="medium_tool",
            permission_level=PermissionLevel.MEDIUM,
            reason="needs it",
        )
        assert manager.request_approval(request) is False

    def test_approval_callback_must_return_bool(self) -> None:
        """The callback contract is a plain bool (enforced by the type system)."""

        def approver(request: ApprovalRequest) -> bool:
            return request.tool_name == "medium_tool"

        manager = PermissionManager(approval=approver)
        granted = ApprovalRequest(
            task_id="t",
            step_id="s",
            tool_name="medium_tool",
            permission_level=PermissionLevel.MEDIUM,
            reason="needs it",
        )
        other = ApprovalRequest(
            task_id="t",
            step_id="s",
            tool_name="other_tool",
            permission_level=PermissionLevel.MEDIUM,
            reason="needs it",
        )
        assert manager.request_approval(granted) is True
        assert manager.request_approval(other) is False
