from agent_core.autonomous import AutonomousLimits, AutonomousRunner
from agent_core.permissions import PermissionLevel
from agent_core.tasks import StepStatus, Task, TaskStep
from agent_core.tools import ToolSpec


class MediumTool:
    spec = ToolSpec(
        name="medium_tool",
        description="test",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        permission_level=PermissionLevel.MEDIUM,
    )

    def run(self, input):
        return None


def test_limits_are_bounded():
    assert AutonomousLimits(max_attempts=3).max_attempts == 3
    try:
        AutonomousLimits(max_attempts=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected bound")


def test_failed_low_risk_task_is_retryable(demo_agent):
    runner = AutonomousRunner(demo_agent)
    task = Task.create("x", now=demo_agent._clock())
    task.steps = [TaskStep(id="1", tool_name="demo_tool", description="x")]
    task.steps[0].transition(StepStatus.RUNNING)
    task.steps[0].transition(StepStatus.FAILED)
    assert runner._can_retry(task)


def test_completed_medium_risk_step_blocks_retry(demo_agent):
    demo_agent.registry.register(MediumTool())
    task = Task.create("x", now=demo_agent._clock())
    task.steps = [TaskStep(id="1", tool_name="medium_tool", description="x")]
    task.steps[0].transition(StepStatus.RUNNING)
    task.steps[0].transition(StepStatus.COMPLETED)
    assert not AutonomousRunner(demo_agent)._can_retry(task)
