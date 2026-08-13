import { Activity, CircleDot, GitBranch } from "lucide-react";

import type { GraphResponse } from "../types";

export function StatusBar({ graph, localMode }: { graph: GraphResponse; localMode: boolean }) {
  return (
    <footer className="status-bar">
      <span><CircleDot size={11} />{graph.nodes.length.toLocaleString()} nodes</span>
      <span><GitBranch size={11} />{graph.edges.length.toLocaleString()} relationships</span>
      <span><Activity size={11} />{localMode ? "Local graph" : graph.truncated ? "Bounded summary" : "Complete slice"}</span>
      <span className="status-bar__snapshot">snapshot {graph.snapshot_id.slice(0, 8)}</span>
    </footer>
  );
}
