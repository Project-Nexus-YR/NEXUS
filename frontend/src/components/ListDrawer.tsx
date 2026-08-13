import {
  AlertTriangle,
  Bot,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Database,
  FlaskConical,
  LoaderCircle,
  PanelLeftClose,
  Play,
  Search,
  Settings,
} from "lucide-react";
import { useState } from "react";

import { nexusApi } from "../api/client";
import type { GraphNode, InvestigationSummary, RuntimeOverview, SidebarView } from "../types";

interface ListDrawerProps {
  view: SidebarView;
  gaps: Record<string, unknown>[];
  contradictions: Record<string, unknown>[];
  investigations: InvestigationSummary[];
  runtime: RuntimeOverview | null;
  sources: GraphNode[];
  pollInterval: number;
  onPollInterval: (value: number) => void;
  onClose: () => void;
  onSelectNode: (id: string) => void;
  onRefresh: () => void;
  onSearch: () => void;
}

export function ListDrawer(props: ListDrawerProps) {
  if (props.view === "graph") return null;
  const title = {
    search: "Search",
    investigations: "Investigations",
    gaps: "Knowledge gaps",
    contradictions: "Contradictions",
    sources: "Sources",
    runtime: "Runtime / Agents",
    settings: "Settings",
    graph: "Graph",
  }[props.view];
  return (
    <section className="list-drawer">
      <header><div><span>Workspace</span><h2>{title}</h2></div><button onClick={props.onClose} title="Close panel"><PanelLeftClose size={16} /></button></header>
      <div className="list-drawer__content">
        {props.view === "search" && <SearchPrompt onSearch={props.onSearch} />}
        {props.view === "gaps" && <GapList {...props} />}
        {props.view === "contradictions" && <ContradictionList {...props} />}
        {props.view === "investigations" && <InvestigationList {...props} />}
        {props.view === "sources" && <SourceList {...props} />}
        {props.view === "runtime" && <RuntimePanel runtime={props.runtime} />}
        {props.view === "settings" && <SettingsPanel pollInterval={props.pollInterval} onPollInterval={props.onPollInterval} />}
      </div>
    </section>
  );
}

function SearchPrompt({ onSearch }: { onSearch: () => void }) {
  return <div className="drawer-empty"><Search size={25} /><strong>Graph-aware search</strong><p>Search entities, claims, documents, sources, gaps, and investigations. Results focus and select knowledge in place.</p><button className="primary-button" onClick={onSearch}>Focus search <kbd>⌘ K</kbd></button></div>;
}

function GapList(props: ListDrawerProps) {
  const [creating, setCreating] = useState<string | null>(null);
  const [actionError, setActionError] = useState<{ id: string; message: string } | null>(null);
  const investigate = async (id: string) => {
    setCreating(id);
    setActionError(null);
    try {
      await nexusApi.investigateGap(id);
      props.onRefresh();
    } catch (reason) {
      setActionError({
        id,
        message: reason instanceof Error ? reason.message : "Unable to create investigation",
      });
    } finally {
      setCreating(null);
    }
  };
  if (!props.gaps.length) return <DrawerEmpty icon={CheckCircle2} title="No unresolved gaps" text="The current knowledge slice has no measurable gaps." />;
  return <div className="drawer-list">{props.gaps.map((gap) => {
    const id = String(gap.id);
    const uncertainty = Number(gap.uncertainty ?? 0);
    const priority = Number(gap.priority ?? 0);
    const sessions = Array.isArray(gap.sessions) ? gap.sessions.length : 0;
    return <article className="drawer-card gap-card" key={id}>
      <button className="drawer-card__main" onClick={() => props.onSelectNode(id)}>
        <div className="drawer-card__icon"><CircleHelp size={16} /></div>
        <div><span className="eyebrow">{String(gap.kind).replaceAll("_", " ")}</span><h3>{String(gap.description)}</h3><p>{String(gap.reason)}</p></div>
        <ChevronRight size={15} />
      </button>
      <div className="card-metrics"><span>uncertainty <strong>{Math.round(uncertainty * 100)}%</strong></span><span>priority <strong>{Math.round(priority * 100)}%</strong></span>{sessions > 0 && <span>{sessions} session{sessions === 1 ? "" : "s"}</span>}</div>
      <button className="card-action" disabled={creating === id} onClick={() => void investigate(id)}>{creating === id ? <LoaderCircle className="spin" size={13} /> : <Play size={13} />}Investigate</button>
      {actionError?.id === id && <p className="card-action-error">{actionError.message}</p>}
    </article>;
  })}</div>;
}

function ContradictionList(props: ListDrawerProps) {
  if (!props.contradictions.length) return <DrawerEmpty icon={CheckCircle2} title="No contradictions" text="No unresolved conflicting claims are present." />;
  return <div className="drawer-list">{props.contradictions.map((item) => <article className="drawer-card contradiction-card" key={String(item.id)}>
    <button className="drawer-card__main" onClick={() => props.onSelectNode(String(item.claim_a_id))}>
      <div className="drawer-card__icon"><AlertTriangle size={16} /></div>
      <div><span className="eyebrow">{String(item.kind).replaceAll("_", " ")}</span><h3>{String(item.description)}</h3><p>Strength {Math.round(Number(item.strength ?? 0) * 100)}%</p></div><ChevronRight size={15} />
    </button>
    <div className="conflict-sides"><button onClick={() => props.onSelectNode(String(item.claim_a_id))}>Claim A</button><i /><button onClick={() => props.onSelectNode(String(item.claim_b_id))}>Claim B</button></div>
  </article>)}</div>;
}

function InvestigationList(props: ListDrawerProps) {
  if (!props.investigations.length) return <DrawerEmpty icon={FlaskConical} title="No investigations" text="Create an investigation from a knowledge gap to begin a durable research session." />;
  return <div className="drawer-list">{props.investigations.map((item) => {
    const active = !["COMPLETED", "FAILED", "CANCELLED"].includes(item.state);
    return <button className="drawer-card investigation-card" key={item.session_id} onClick={() => props.onSelectNode(item.session_id)}>
      <div className="investigation-card__head"><span className={`run-indicator ${active ? "run-indicator--active" : ""}`} /><span>{item.phase}</span><em>iteration {item.iteration}</em></div>
      <h3>{item.question}</h3>
      <InvestigationStages state={item.state} />
      <div className="card-metrics">{Object.entries(item.task_counts).map(([state, count]) => <span key={state}>{state.toLowerCase()} <strong>{count}</strong></span>)}</div>
    </button>;
  })}</div>;
}

function InvestigationStages({ state }: { state: string }) {
  const stages = ["PLANNING", "EXECUTING", "EVALUATING", "UPDATING", "COMPLETED"];
  const index = Math.max(0, stages.indexOf(state));
  return <div className="run-stages">{stages.map((stage, stageIndex) => <span key={stage} className={stageIndex <= index ? "is-done" : ""} title={stage} />)}</div>;
}

function SourceList(props: ListDrawerProps) {
  if (!props.sources.length) return <DrawerEmpty icon={Database} title="No sources" text="Ingest a knowledge snapshot to inspect source provenance." />;
  return <div className="drawer-list">{props.sources.map((source) => <button className="drawer-card source-card" key={source.id} onClick={() => props.onSelectNode(source.id)}><div className="drawer-card__icon"><Database size={15} /></div><div><span className="eyebrow">{source.subtitle}</span><h3>{source.label}</h3><p>{String(source.metadata.reference ?? "No external reference")}</p></div><ChevronRight size={15} /></button>)}</div>;
}

function RuntimePanel({ runtime }: { runtime: RuntimeOverview | null }) {
  if (!runtime) return <div className="drawer-empty"><span className="spinner" />Loading runtime…</div>;
  return <div className="runtime-panel">
    <div className="runtime-summary"><Bot size={20} /><div><strong>{runtime.available ? "Runtime connected" : "Runtime unavailable"}</strong><span>{runtime.tasks.length} durable task{runtime.tasks.length === 1 ? "" : "s"}</span></div><i className={runtime.available ? "is-online" : ""} /></div>
    <section><h3>Queue state</h3><div className="runtime-grid">{Object.entries(runtime.queue).map(([state, count]) => <div key={state}><strong>{count}</strong><span>{state.replaceAll("_", " ")}</span></div>)}</div></section>
    <section><h3>Workers</h3>{runtime.workers.length === 0 ? <div className="inline-empty">No workers are registered in this coordinator process.</div> : runtime.workers.map((worker) => <div className="worker-row" key={String(worker.worker_id)}><span className="run-indicator run-indicator--active" /><div><strong>{String(worker.worker_id)}</strong><small>{String(worker.status)} · {String(worker.available_slots)} slots</small></div></div>)}</section>
    <section><h3>Recent tasks</h3>{runtime.tasks.slice(0, 12).map((task) => <div className="task-row" key={String(task.task_id)}><i /><div><strong>{String(task.run_id)}</strong><small>{String(task.state)} · attempt {String(task.attempt)}</small></div></div>)}</section>
  </div>;
}

function SettingsPanel({ pollInterval, onPollInterval }: { pollInterval: number; onPollInterval: (value: number) => void }) {
  return <div className="settings-panel"><div className="settings-intro"><Settings size={20} /><div><strong>Explorer settings</strong><span>Presentation preferences are local to this browser.</span></div></div><label><span><strong>Live update polling</strong><small>Refresh graph and runtime from the real backend.</small></span><select value={pollInterval} onChange={(event) => onPollInterval(Number(event.target.value))}><option value={0}>Off</option><option value={5000}>5 seconds</option><option value={8000}>8 seconds</option><option value={15000}>15 seconds</option><option value={30000}>30 seconds</option></select></label><div className="settings-note">NEXUS uses bounded server-side graph slices. Increase the graph limit in filters only when your machine can comfortably render the additional elements.</div></div>;
}

function DrawerEmpty({ icon: Icon, title, text }: { icon: typeof CircleHelp; title: string; text: string }) {
  return <div className="drawer-empty"><Icon size={25} /><strong>{title}</strong><p>{text}</p></div>;
}
