"""Event bus and structured event tests (deterministic via injected clock)."""

from __future__ import annotations

from agent_core import AgentEvent, EventBus, EventType
from conftest import FIXED_NOW


class TestEventBus:
    def test_emit_records_event(self) -> None:
        bus = EventBus(clock=lambda: FIXED_NOW)
        event = bus.emit(EventType.TASK_CREATED, task_id="t-1", data={"request": "hi"})
        assert len(bus) == 1
        assert bus.history[0] is event
        assert event.type is EventType.TASK_CREATED
        assert event.task_id == "t-1"
        assert event.timestamp == FIXED_NOW
        assert event.step_id is None

    def test_to_dict_is_stable_and_serializable(self) -> None:
        import json

        bus = EventBus(clock=lambda: FIXED_NOW)
        event = bus.emit(EventType.TOOL_STARTED, task_id="t", step_id="s", data={"tool_name": "x"})
        payload = event.to_dict()
        assert set(payload) == {"event_id", "type", "task_id", "step_id", "timestamp", "data"}
        assert payload["type"] == "TOOL_STARTED"
        assert payload["timestamp"] == FIXED_NOW.isoformat()
        json.dumps(payload)

    def test_subscribers_receive_events_in_order(self) -> None:
        bus = EventBus(clock=lambda: FIXED_NOW)
        seen: list[EventType] = []
        bus.subscribe(lambda e: seen.append(e.type))
        bus.emit(EventType.TASK_CREATED, task_id="t")
        bus.emit(EventType.TASK_COMPLETED, task_id="t")
        assert seen == [EventType.TASK_CREATED, EventType.TASK_COMPLETED]

    def test_multiple_subscribers_all_notified(self) -> None:
        bus = EventBus(clock=lambda: FIXED_NOW)
        counts = [0, 0]

        def first(event: AgentEvent) -> None:
            counts[0] += 1

        def second(event: AgentEvent) -> None:
            counts[1] += 1

        bus.subscribe(first)
        bus.subscribe(second)
        bus.emit(EventType.TASK_CREATED)
        assert counts == [1, 1]

    def test_events_of_type_filters(self) -> None:
        bus = EventBus(clock=lambda: FIXED_NOW)
        bus.emit(EventType.TASK_CREATED, task_id="a")
        bus.emit(EventType.TOOL_FAILED, task_id="a")
        bus.emit(EventType.TASK_FAILED, task_id="a")
        assert [e.task_id for e in bus.events_of_type(EventType.TOOL_FAILED)] == ["a"]
        assert len(bus.events_of_type(EventType.TOOL_STARTED)) == 0

    def test_emit_copies_caller_data(self) -> None:
        bus = EventBus(clock=lambda: FIXED_NOW)
        data = {"k": "v"}
        bus.emit(EventType.TASK_CREATED, data=data)
        data["k"] = "mutated"
        assert bus.history[0].data == {"k": "v"}

    def test_event_data_is_operational_only(self) -> None:
        """Event payloads carry bounded operational facts, not CoT or secrets."""
        bus = EventBus(clock=lambda: FIXED_NOW)
        event = bus.emit(
            EventType.TOOL_COMPLETED,
            task_id="t",
            data={"tool_name": "demo_tool", "output": {"message": "ok"}},
        )
        assert "tool_name" in event.data
        assert event.data["output"] == {"message": "ok"}


def test_event_bus_redacts_sensitive_keys() -> None:
    bus = EventBus()
    event = bus.emit(
        EventType.TASK_CREATED,
        task_id="t1",
        data={"api_key": "secret", "nested": {"password": "pw", "safe": "ok"}},
    )
    assert event.data["api_key"] == {"redacted": True}
    assert event.data["nested"]["password"] == {"redacted": True}
    assert event.data["nested"]["safe"] == "ok"
