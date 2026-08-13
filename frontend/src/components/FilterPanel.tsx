import { ChevronDown, SlidersHorizontal } from "lucide-react";
import { useState } from "react";

import type { GraphFilters, NodeKind } from "../types";

interface FilterPanelProps {
  filters: GraphFilters;
  relationTypes: string[];
  onChange: (filters: GraphFilters) => void;
}

const KINDS: Array<{ kind: NodeKind; label: string }> = [
  { kind: "entity", label: "Entities" },
  { kind: "claim", label: "Claims" },
  { kind: "gap", label: "Gaps" },
  { kind: "investigation", label: "Investigations" },
  { kind: "evidence", label: "Evidence" },
  { kind: "document", label: "Documents" },
  { kind: "source", label: "Sources" },
];

export function FilterPanel({ filters, relationTypes, onChange }: FilterPanelProps) {
  const [open, setOpen] = useState(true);
  const updateKind = (kind: NodeKind) => {
    const nodeKinds = filters.nodeKinds.includes(kind)
      ? filters.nodeKinds.filter((item) => item !== kind)
      : [...filters.nodeKinds, kind];
    onChange({ ...filters, nodeKinds });
  };
  const sourceEvidenceVisible = ["source", "document", "evidence"].some((kind) => filters.nodeKinds.includes(kind as NodeKind));
  const contradictionOnly = filters.relationTypes.length === 1 && filters.relationTypes[0] === "contradiction";
  const gapOnly = filters.relationTypes.includes("knowledge_gap") && filters.nodeKinds.includes("gap");
  const updateRelationType = (kind: string) => {
    const next = filters.relationTypes.includes(kind)
      ? filters.relationTypes.filter((item) => item !== kind)
      : [...filters.relationTypes, kind];
    onChange({ ...filters, relationTypes: next });
  };

  return (
    <section className={`filter-panel ${open ? "filter-panel--open" : ""}`}>
      <button className="filter-panel__header" onClick={() => setOpen((value) => !value)}>
        <SlidersHorizontal size={15} /><span>Graph filters</span><ChevronDown size={14} />
      </button>
      {open && (
        <div className="filter-panel__body">
          <div className="filter-label">Node types</div>
          <div className="kind-filters">
            {KINDS.map(({ kind, label }) => (
              <label key={kind}>
                <input type="checkbox" checked={filters.nodeKinds.includes(kind)} onChange={() => updateKind(kind)} />
                <span className={`kind-dot kind-dot--${kind}`} /> {label}
              </label>
            ))}
          </div>
          {relationTypes.length > 0 && (
            <>
              <div className="filter-label filter-label--row">
                <span>Edge types</span>
                {filters.relationTypes.length > 0 && (
                  <button onClick={() => onChange({ ...filters, relationTypes: [] })}>Clear</button>
                )}
              </div>
              <div className="edge-filters">
                {relationTypes.slice(0, 12).map((kind) => (
                  <label key={kind} title={kind}>
                    <input
                      type="checkbox"
                      checked={filters.relationTypes.includes(kind)}
                      onChange={() => updateRelationType(kind)}
                    />
                    <span>{kind.replaceAll("_", " ")}</span>
                  </label>
                ))}
              </div>
            </>
          )}
          <div className="filter-label filter-label--row">
            <span>Minimum confidence</span><strong>{Math.round(filters.minConfidence * 100)}%</strong>
          </div>
          <input
            className="range"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={filters.minConfidence}
            onChange={(event) => onChange({ ...filters, minConfidence: Number(event.target.value) })}
          />
          <div className="quick-filters">
            <button
              className={!sourceEvidenceVisible ? "is-active" : ""}
              onClick={() => onChange({
                ...filters,
                nodeKinds: sourceEvidenceVisible
                  ? filters.nodeKinds.filter((kind) => !["source", "document", "evidence"].includes(kind))
                  : Array.from(new Set([...filters.nodeKinds, "source", "document", "evidence"])),
              })}
            >Hide provenance nodes</button>
            <button
              className={contradictionOnly ? "is-active" : ""}
              onClick={() => onChange({ ...filters, relationTypes: contradictionOnly ? [] : ["contradiction"], nodeKinds: ["claim"] })}
            >Contradictions only</button>
            <button
              className={gapOnly ? "is-active" : ""}
              onClick={() => onChange({
                ...filters,
                relationTypes: gapOnly ? [] : ["knowledge_gap", "investigation_target"],
                nodeKinds: gapOnly ? KINDS.map((item) => item.kind) : ["gap", "entity", "claim", "investigation"],
              })}
            >Gaps only</button>
          </div>
        </div>
      )}
    </section>
  );
}
