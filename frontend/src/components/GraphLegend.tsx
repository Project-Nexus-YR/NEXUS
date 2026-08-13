import type { NodeKind } from "../types";

const ITEMS: Array<[NodeKind, string]> = [
  ["entity", "Entity"],
  ["claim", "Claim"],
  ["gap", "Unresolved gap"],
  ["investigation", "Investigation"],
  ["evidence", "Evidence"],
  ["source", "Source"],
];

export function GraphLegend() {
  return (
    <div className="graph-legend">
      {ITEMS.map(([kind, label]) => <span key={kind}><i className={`kind-dot kind-dot--${kind}`} />{label}</span>)}
      <span><i className="legend-line legend-line--contradiction" />Conflict</span>
      <span><i className="legend-line legend-line--provenance" />Provenance</span>
    </div>
  );
}
