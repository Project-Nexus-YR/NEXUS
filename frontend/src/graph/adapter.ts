import Graph from "graphology";
import type { Attributes } from "graphology-types";

import type { GraphEdge, GraphNode, GraphResponse, NodeKind } from "../types";

export interface VisualNodeAttributes extends Attributes {
  x: number;
  y: number;
  size: number;
  color: string;
  label: string;
  kind: NodeKind;
  raw: GraphNode;
  highlighted?: boolean;
  forceLabel?: boolean;
}

export interface VisualEdgeAttributes extends Attributes {
  size: number;
  color: string;
  label: string;
  kind: string;
  type: "line" | "arrow";
  raw: GraphEdge;
}

const NODE_COLORS: Record<NodeKind, string> = {
  entity: "#a7b0bd",
  claim: "#66a6b8",
  evidence: "#758398",
  document: "#8a7fa8",
  source: "#706e82",
  gap: "#d9a451",
  investigation: "#52c7a5",
};

export function nodeColor(node: GraphNode): string {
  if (node.verification_state === "contradicted" || node.status === "contradicted") return "#e36f6f";
  if (node.kind === "claim" && node.verification_state === "verified") return "#6db8a6";
  return NODE_COLORS[node.kind];
}

export function edgeColor(edge: GraphEdge): string {
  if (edge.kind === "contradiction" || edge.kind === "contradict") return "#d96262";
  if (edge.kind === "support") return "#55a88e";
  if (edge.kind === "knowledge_gap") return "#b98b48";
  if (edge.kind === "investigation_target") return "#42a98e";
  if (edge.kind === "provenance") return "#554f68";
  return "#35404c";
}

export function buildGraph(data: GraphResponse): Graph<VisualNodeAttributes, VisualEdgeAttributes> {
  const graph = new Graph<VisualNodeAttributes, VisualEdgeAttributes>({ multi: true, type: "directed" });
  data.nodes.forEach((node) => {
    const [x, y] = stablePosition(node.id);
    graph.addNode(node.id, {
      x,
      y,
      size: nodeSize(node),
      color: nodeColor(node),
      label: node.label,
      kind: node.kind,
      raw: node,
      forceLabel: node.kind === "investigation",
    });
  });
  data.edges.forEach((edge) => {
    if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) return;
    graph.addDirectedEdgeWithKey(edge.id, edge.source, edge.target, {
      size: edge.kind === "contradiction" ? 2.2 : Math.max(0.4, (edge.confidence ?? 0.5) * 1.5),
      color: edgeColor(edge),
      label: edge.label,
      kind: edge.kind,
      type: edge.directed ? "arrow" : "line",
      raw: edge,
    });
  });
  return graph;
}

export function nodeSize(node: GraphNode): number {
  const base: Record<NodeKind, number> = {
    entity: 5.2,
    claim: 4.4,
    evidence: 2.8,
    document: 3.8,
    source: 4.2,
    gap: 6.2,
    investigation: 6.6,
  };
  const connectivity = Math.min(5, Math.log2(node.degree + 1) * 1.25);
  const weight = (node.importance ?? 0) * 2 + (node.confidence ?? 0) * 0.8;
  return base[node.kind] + connectivity + weight;
}

function stablePosition(id: string): [number, number] {
  let first = 2166136261;
  let second = 2246822519;
  for (let index = 0; index < id.length; index += 1) {
    first = Math.imul(first ^ id.charCodeAt(index), 16777619);
    second = Math.imul(second ^ id.charCodeAt(id.length - index - 1), 3266489917);
  }
  const angle = ((first >>> 0) / 4294967295) * Math.PI * 2;
  const radius = 0.25 + ((second >>> 0) / 4294967295) * 0.75;
  return [Math.cos(angle) * radius, Math.sin(angle) * radius];
}
