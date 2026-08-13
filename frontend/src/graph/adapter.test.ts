import { describe, expect, it } from "vitest";

import type { GraphNode, GraphResponse } from "../types";
import { buildGraph, edgeColor, nodeColor, nodeSize } from "./adapter";

function node(id: string, kind: GraphNode["kind"], overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    id,
    kind,
    label: id,
    subtitle: kind,
    confidence: null,
    uncertainty: null,
    importance: null,
    verification_state: null,
    status: null,
    degree: 0,
    created_at: null,
    updated_at: null,
    metadata: {},
    ...overrides,
  };
}

describe("graph adapter", () => {
  it("preserves backend identifiers and drops dangling edges", () => {
    const data: GraphResponse = {
      nodes: [node("ada", "entity"), node("claim-1", "claim", { confidence: 0.8 })],
      edges: [
        { id: "edge-1", source: "ada", target: "claim-1", kind: "claim_subject", label: "asserts", confidence: 0.8, verification_state: null, directed: true, metadata: {} },
        { id: "dangling", source: "ada", target: "missing", kind: "relation", label: "bad", confidence: null, verification_state: null, directed: true, metadata: {} },
      ],
      total_nodes: 2,
      total_edges: 2,
      truncated: false,
      snapshot_id: "snapshot",
      statistics: {},
    };

    const graph = buildGraph(data);

    expect(graph.nodes()).toEqual(["ada", "claim-1"]);
    expect(graph.edges()).toEqual(["edge-1"]);
    expect(graph.getNodeAttribute("claim-1", "raw").confidence).toBe(0.8);
    expect(graph.getEdgeAttribute("edge-1", "type")).toBe("arrow");
  });

  it("uses epistemic state and graph importance in styling", () => {
    const contradicted = node("claim", "claim", { verification_state: "contradicted" });
    const verified = node("verified", "claim", { verification_state: "verified" });
    const centralGap = node("gap", "gap", { degree: 12, importance: 0.9, uncertainty: 0.8 });

    expect(nodeColor(contradicted)).toBe("#e36f6f");
    expect(nodeColor(verified)).toBe("#6db8a6");
    expect(nodeSize(centralGap)).toBeGreaterThan(nodeSize(node("plain", "gap")));
    expect(edgeColor({ id: "c", source: "a", target: "b", kind: "contradiction", label: "conflicts", confidence: null, verification_state: null, directed: false, metadata: {} })).toBe("#d96262");
  });

  it("starts nodes at stable deterministic positions", () => {
    const response: GraphResponse = {
      nodes: [node("stable", "entity")],
      edges: [],
      total_nodes: 1,
      total_edges: 0,
      truncated: false,
      snapshot_id: "one",
      statistics: {},
    };
    const first = buildGraph(response).getNodeAttributes("stable");
    const second = buildGraph({ ...response, snapshot_id: "two" }).getNodeAttributes("stable");

    expect([first.x, first.y]).toEqual([second.x, second.y]);
  });
});
