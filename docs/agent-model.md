# Agent model

`Agent`, `AgentRun`, `AgentStep`, `Observation`, and `ToolCall` have opaque stable IDs.
An Agent role is data: capabilities, preferred tools, constraints, instructions, and
output schema.
The built-in role names are Planner, Researcher, Experimenter, Analyst, Critic, and
Synthesizer; adding a role does not change scheduler or loop code.

Context is held as artifact references plus bounded summaries. Raw tool payloads should
be stored by an artifact adapter and referenced from observations/checkpoints, avoiding
unbounded AgentRun rows. Budgets constrain tokens, wall time, tool calls, worker count,
and experiment resources; exhausted loop budgets pause and checkpoint the run.
