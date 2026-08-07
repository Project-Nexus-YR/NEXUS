"""Safe inspection/replay: reconstructs a run without re-executing side effects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .events import Event


@dataclass(slots=True)
class ReplayView:
    trace_id: str
    events: list[Event] = field(default_factory=list)
    task_states: dict[str, str] = field(default_factory=dict)
    agent_states: dict[str, str] = field(default_factory=dict)
    tool_calls: list[str] = field(default_factory=list)


class RunReplayer:
    def reconstruct(self, trace_id: str, events: Iterable[Event]) -> ReplayView:
        view = ReplayView(trace_id)
        for event in events:
            if event.trace_id != trace_id:
                continue
            view.events.append(event)
            task_id = event.payload.get("task_id")
            if isinstance(task_id, str) and "state" in event.payload:
                view.task_states[task_id] = str(event.payload["state"])
            run_id = event.payload.get("run_id")
            if isinstance(run_id, str) and event.event_type == "agent.run.transitioned":
                view.agent_states[run_id] = str(event.payload["state"])
            tool_call_id = event.payload.get("tool_call_id")
            if isinstance(tool_call_id, str):
                view.tool_calls.append(tool_call_id)
        return view
