from __future__ import annotations

from threading import Thread
from time import sleep

from agent_core.approval import ApprovalBroker
from agent_core.permissions import ApprovalRequest, PermissionLevel


def request() -> ApprovalRequest:
    return ApprovalRequest(
        task_id="task-1",
        step_id="step-1",
        tool_name="write_text_file",
        permission_level=PermissionLevel.MEDIUM,
        reason="write a file",
    )


def test_approval_broker_blocks_until_decision() -> None:
    broker = ApprovalBroker(timeout_s=2)
    result: list[bool] = []
    thread = Thread(target=lambda: result.append(broker.request(request())), daemon=True)
    thread.start()
    for _ in range(20):
        if broker.has_pending("task-1", "step-1"):
            break
        sleep(0.01)
    assert broker.decide("task-1", "step-1", True)
    thread.join(timeout=1)
    assert result == [True]
    assert broker.list_pending() == []


def test_approval_broker_fails_closed_on_unknown_decision() -> None:
    broker = ApprovalBroker(timeout_s=0.05)
    assert broker.decide("missing", "step", True) is False
    assert broker.request(request()) is False
