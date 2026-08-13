export type NodeKind =
  | "entity"
  | "claim"
  | "evidence"
  | "document"
  | "source"
  | "gap"
  | "investigation";

export interface GraphNode {
  id: string;
  kind: NodeKind;
  label: string;
  subtitle: string;
  confidence: number | null;
  uncertainty: number | null;
  importance: number | null;
  verification_state: string | null;
  status: string | null;
  degree: number;
  created_at: string | null;
  updated_at: string | null;
  metadata: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  kind: string;
  label: string;
  confidence: number | null;
  verification_state: string | null;
  directed: boolean;
  metadata: Record<string, unknown>;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  total_nodes: number;
  total_edges: number;
  truncated: boolean;
  snapshot_id: string;
  statistics: Record<string, number>;
}

export interface Connection {
  edge: GraphEdge;
  node: Pick<GraphNode, "id" | "kind" | "label" | "subtitle">;
}

export interface NodeDetail {
  node: GraphNode;
  attributes: Record<string, unknown>;
  connections: Connection[];
  evidence: Record<string, unknown>[];
  provenance: Record<string, unknown>;
  contradictions: Record<string, unknown>[];
  gaps: Record<string, unknown>[];
  history: Record<string, unknown>[];
}

export interface SearchResult extends GraphNode {
  score: number;
}

export interface GraphFilters {
  nodeKinds: NodeKind[];
  relationTypes: string[];
  verificationStates: string[];
  minConfidence: number;
  maxNodes: number;
  maxEdges: number;
}

export interface InvestigationSummary {
  session_id: string;
  objective_id: string;
  question: string;
  state: string;
  phase: string;
  iteration: number;
  created_at: string;
  updated_at: string;
  termination_reason: string | null;
  remaining_budget: Record<string, number>;
  task_counts: Record<string, number>;
  target_gap_ids: string[];
  progress: Record<string, unknown> | null;
  verification: Record<string, unknown> | null;
  evidence: Record<string, unknown> | null;
}

export interface RuntimeOverview {
  available: boolean;
  queue: Record<string, number>;
  statistics: Record<string, unknown>;
  workers: Record<string, unknown>[];
  tasks: Record<string, unknown>[];
}

export type SidebarView =
  | "graph"
  | "search"
  | "investigations"
  | "gaps"
  | "contradictions"
  | "sources"
  | "runtime"
  | "settings";

export interface Selection {
  type: "node" | "edge";
  id: string;
}
