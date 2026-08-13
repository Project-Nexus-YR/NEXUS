import { afterEach, describe, expect, it, vi } from "vitest";

import type { GraphFilters } from "../types";
import { nexusApi } from "./client";

const filters: GraphFilters = {
  nodeKinds: ["entity", "claim"],
  relationTypes: ["works_at"],
  verificationStates: ["supported"],
  minConfidence: 0.4,
  maxNodes: 120,
  maxEdges: 300,
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("NEXUS API client", () => {
  it("serializes graph filters for server-side slicing", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ nodes: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }));
    vi.stubGlobal("fetch", fetchMock);

    await nexusApi.graph(filters);

    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain("kinds=entity");
    expect(url).toContain("kinds=claim");
    expect(url).toContain("relation_types=works_at");
    expect(url).toContain("verification_states=supported");
    expect(url).toContain("max_nodes=120");
  });

  it("encodes local graph identifiers and depth", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ nodes: [] }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await nexusApi.neighborhood("Ada Lovelace/claim", 3, filters);

    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain("/api/graph/neighborhood/Ada%20Lovelace%2Fclaim");
    expect(url).toContain("depth=3");
  });

  it("surfaces backend and network failures as typed errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: "unknown graph node" }),
      { status: 404, headers: { "Content-Type": "application/json" } },
    )));
    await expect(nexusApi.node("missing")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      message: "unknown graph node",
    });

    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    await expect(nexusApi.runtime()).rejects.toMatchObject({
      status: 0,
      message: expect.stringContaining("offline"),
    });
  });
});
