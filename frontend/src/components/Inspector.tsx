import { ChevronRight, ExternalLink, Focus, PanelRightClose, X } from "lucide-react";
import { useState } from "react";

import type { GraphEdge, NodeDetail } from "../types";

type InspectorTab = "overview" | "evidence" | "connections" | "provenance" | "history";

interface InspectorProps {
  detail: NodeDetail | null;
  edge: GraphEdge | null;
  loading: boolean;
  error: Error | null;
  collapsed: boolean;
  localMode: boolean;
  onCollapse: () => void;
  onClose: () => void;
  onFocusNode: (nodeId: string) => void;
  onLocalGraph: () => void;
}

const TABS: Array<[InspectorTab, string]> = [
  ["overview", "Overview"],
  ["evidence", "Evidence"],
  ["connections", "Connections"],
  ["provenance", "Provenance"],
  ["history", "History"],
];

export function Inspector(props: InspectorProps) {
  const [tab, setTab] = useState<InspectorTab>("overview");
  if (props.collapsed) {
    return <button className="inspector-peek" onClick={props.onCollapse} aria-label="Open inspector"><ChevronRight size={17} /></button>;
  }
  return (
    <aside className="inspector">
      <div className="inspector__topbar">
        <span>Inspector</span>
        <div><button onClick={props.onCollapse} title="Collapse inspector"><PanelRightClose size={15} /></button><button onClick={props.onClose} title="Clear selection"><X size={15} /></button></div>
      </div>
      {props.loading && <div className="inspector-state"><span className="spinner" />Reading knowledge…</div>}
      {props.error && <div className="inspector-state inspector-state--error">{props.error.message}</div>}
      {!props.loading && !props.detail && !props.edge && (
        <div className="inspector-empty"><Focus size={28} /><strong>Select graph knowledge</strong><p>Click a node or relationship to inspect its evidence, provenance, uncertainty, and history.</p></div>
      )}
      {props.edge && !props.detail && <EdgeInspector edge={props.edge} />}
      {props.detail && (
        <>
          <header className="inspector__header">
            <span className={`kind-chip kind-chip--${props.detail.node.kind}`}>{props.detail.node.kind}</span>
            <h2>{props.detail.node.label}</h2>
            <p>{props.detail.node.subtitle || "NEXUS knowledge object"}</p>
            <div className="metric-strip">
              {props.detail.node.confidence !== null && <Metric label="Confidence" value={props.detail.node.confidence} />}
              {props.detail.node.uncertainty !== null && <Metric label="Uncertainty" value={props.detail.node.uncertainty} warning />}
              {props.detail.node.status && <div className="state-pill">{props.detail.node.status}</div>}
            </div>
            <button className={`local-graph-button ${props.localMode ? "is-active" : ""}`} onClick={props.onLocalGraph}><Focus size={14} />{props.localMode ? "Return to global graph" : "Show local graph"}</button>
          </header>
          <div className="inspector-tabs">
            {TABS.map(([id, label]) => <button key={id} className={tab === id ? "is-active" : ""} onClick={() => setTab(id)}>{label}</button>)}
          </div>
          <div className="inspector__content">
            {tab === "overview" && <Overview detail={props.detail} />}
            {tab === "evidence" && <CardList items={props.detail.evidence} empty="No evidence is attached to this object." />}
            {tab === "connections" && (
              <div className="connection-list">
                {props.detail.connections.length === 0 && <Empty text="No visible graph connections." />}
                {props.detail.connections.map(({ edge, node }) => (
                  <button key={edge.id} onClick={() => props.onFocusNode(node.id)}>
                    <span className={`kind-dot kind-dot--${node.kind}`} />
                    <span><strong>{node.label}</strong><small>{edge.label} · {node.kind}</small></span>
                    <ChevronRight size={14} />
                  </button>
                ))}
              </div>
            )}
            {tab === "provenance" && <ObjectView value={props.detail.provenance} empty="No provenance is available." />}
            {tab === "history" && <CardList items={props.detail.history} empty="No lifecycle history is available." />}
          </div>
        </>
      )}
    </aside>
  );
}

function Overview({ detail }: { detail: NodeDetail }) {
  return (
    <div className="overview-stack">
      <section><h3>Properties</h3><ObjectView value={detail.attributes} /></section>
      {detail.contradictions.length > 0 && <section className="warning-section"><h3>Contradictions</h3><CardList items={detail.contradictions} /></section>}
      {detail.gaps.length > 0 && <section><h3>Knowledge gaps</h3><CardList items={detail.gaps} /></section>}
    </div>
  );
}

function EdgeInspector({ edge }: { edge: GraphEdge }) {
  return (
    <div className="edge-inspector">
      <span className={`kind-chip ${edge.kind === "contradiction" ? "kind-chip--warning" : ""}`}>relationship</span>
      <h2>{edge.label}</h2><p>{edge.kind.replaceAll("_", " ")}</p>
      <section><h3>Endpoints</h3><InfoRow label="From" value={edge.source} /><InfoRow label="To" value={edge.target} /></section>
      {edge.confidence !== null && <Metric label="Confidence" value={edge.confidence} />}
      <section><h3>Metadata</h3><ObjectView value={edge.metadata} /></section>
    </div>
  );
}

function Metric({ label, value, warning = false }: { label: string; value: number; warning?: boolean }) {
  return <div className={`metric ${warning ? "metric--warning" : ""}`}><span>{label}</span><strong>{Math.round(value * 100)}%</strong><i><b style={{ width: `${Math.round(value * 100)}%` }} /></i></div>;
}

function ObjectView({ value, empty = "No data available." }: { value: Record<string, unknown>; empty?: string }) {
  const entries = Object.entries(value).filter(([, item]) => item !== null && item !== "" && !(Array.isArray(item) && !item.length));
  if (!entries.length) return <Empty text={empty} />;
  return <div className="property-list">{entries.map(([key, item]) => <InfoRow key={key} label={humanize(key)} value={item} />)}</div>;
}

function InfoRow({ label, value }: { label: string; value: unknown }) {
  if (typeof value === "object" && value !== null) {
    return <div className="info-row info-row--stack"><span>{label}</span><pre>{JSON.stringify(value, null, 2)}</pre></div>;
  }
  const display = typeof value === "number" ? Number(value.toFixed(4)).toString() : String(value);
  const link = typeof value === "string" && /^(https?:\/\/|file:|source:)/.test(value);
  return <div className="info-row"><span>{label}</span><strong>{display}</strong>{link && <ExternalLink size={11} />}</div>;
}

function CardList({ items, empty = "No records." }: { items: Record<string, unknown>[]; empty?: string }) {
  if (!items.length) return <Empty text={empty} />;
  return <div className="record-list">{items.map((item, index) => <div className="record-card" key={String(item.id ?? item.artifact_id ?? index)}><ObjectView value={item} /></div>)}</div>;
}

function Empty({ text }: { text: string }) { return <div className="inline-empty">{text}</div>; }
function humanize(value: string) { return value.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase()); }
