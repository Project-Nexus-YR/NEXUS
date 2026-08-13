import type {
  GraphFilters,
  GraphResponse,
  InvestigationSummary,
  NodeDetail,
  RuntimeOverview,
  SearchResult,
} from "../types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json", ...init?.headers },
      ...init,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError(
      error instanceof Error ? `Cannot reach the NEXUS API: ${error.message}` : "Cannot reach the NEXUS API",
      0,
    );
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload && typeof payload === "object" && "detail" in payload ? payload.detail : payload;
    throw new ApiError(typeof detail === "string" ? detail : `NEXUS API returned ${response.status}`, response.status, detail);
  }
  return payload as T;
}

function graphQuery(filters: GraphFilters): URLSearchParams {
  const query = new URLSearchParams({
    min_confidence: String(filters.minConfidence),
    max_nodes: String(filters.maxNodes),
    max_edges: String(filters.maxEdges),
  });
  filters.nodeKinds.forEach((value) => query.append("kinds", value));
  filters.relationTypes.forEach((value) => query.append("relation_types", value));
  filters.verificationStates.forEach((value) => query.append("verification_states", value));
  return query;
}

export const nexusApi = {
  graph(filters: GraphFilters, signal?: AbortSignal): Promise<GraphResponse> {
    return request(`/api/graph?${graphQuery(filters)}`, { signal });
  },

  neighborhood(nodeId: string, depth: number, filters: GraphFilters, signal?: AbortSignal): Promise<GraphResponse> {
    const query = graphQuery(filters);
    query.set("depth", String(depth));
    return request(`/api/graph/neighborhood/${encodeURIComponent(nodeId)}?${query}`, { signal });
  },

  node(nodeId: string, signal?: AbortSignal): Promise<NodeDetail> {
    return request(`/api/nodes/${encodeURIComponent(nodeId)}`, { signal });
  },

  search(query: string, signal?: AbortSignal): Promise<SearchResult[]> {
    return request(`/api/search?q=${encodeURIComponent(query)}&limit=30`, { signal });
  },

  gaps(signal?: AbortSignal): Promise<Record<string, unknown>[]> {
    return request("/api/gaps", { signal });
  },

  contradictions(signal?: AbortSignal): Promise<Record<string, unknown>[]> {
    return request("/api/contradictions", { signal });
  },

  sources(signal?: AbortSignal): Promise<GraphResponse["nodes"]> {
    return request("/api/sources", { signal });
  },

  investigations(signal?: AbortSignal): Promise<InvestigationSummary[]> {
    return request("/api/investigations", { signal });
  },

  runtime(signal?: AbortSignal): Promise<RuntimeOverview> {
    return request("/api/runtime", { signal });
  },

  investigateGap(gapId: string): Promise<InvestigationSummary> {
    return request(`/api/gaps/${encodeURIComponent(gapId)}/investigations`, {
      method: "POST",
      body: JSON.stringify({}),
    });
  },
};
