"""Demo entry point for the agent core.

Runs the full end-to-end flow through the Model Gateway:

    .venv/bin/python apps/backend/src/main.py "Run the demo tool."

Provider selection is environment-driven (see .env.example):

- Default (``MODEL_PROVIDER=mock``): fully offline, no API keys.
- ``MODEL_PROVIDER=openai`` + ``OPENAI_API_KEY`` (and the optional
  ``openai`` extra installed): uses a real model for planning.

The same code path is reused by all future entry points. Exit codes: 0 =
task COMPLETED, 1 = task failed/cancelled, 2 = core-level error (e.g.
provider misconfiguration).
"""

from __future__ import annotations

import argparse
import logging

from agent_core import Agent, AgentCoreError, Settings, build_gateway


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one request through the agent core (provider via MODEL_PROVIDER)."
    )
    parser.add_argument(
        "request",
        nargs="?",
        default="Run the demo tool.",
        help="Natural-language request to run.",
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    settings.configure_logging()
    log = logging.getLogger("apps.backend")
    log.info("agent=%s", settings.agent_name)

    try:
        gateway = build_gateway(settings)
        log.info("provider=%s", gateway.name)
        agent = Agent.create_configured(settings=settings, gateway=gateway)
        task = agent.run(args.request)
    except AgentCoreError as exc:
        # Controlled core-level failure (e.g. provider misconfiguration).
        log.error("%s: %s", type(exc).__name__, exc)
        return 2

    for event in agent.events.history:
        log.info("event: %s %s", event.type.value, event.data)

    print(f"task_id: {task.id}")
    print(f"state:   {task.state.value}")
    print(f"steps:   {len(task.steps)}")
    print(f"result:  {task.result!r}")
    if task.error:
        print(f"error:   {task.error}")
    return 0 if task.state.value == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
