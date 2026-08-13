import { useCallback, useEffect, useRef, useState } from "react";

import { nexusApi } from "../api/client";
import type {
  GraphFilters,
  GraphNode,
  GraphResponse,
  InvestigationSummary,
  NodeDetail,
  RuntimeOverview,
  SearchResult,
} from "../types";

interface GraphQuery {
  filters: GraphFilters;
  localNodeId: string | null;
  localDepth: number;
  pollInterval: number;
}

export function useGraphData({ filters, localNodeId, localDepth, pollInterval }: GraphQuery) {
  const [data, setData] = useState<GraphResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const [newNodeIds, setNewNodeIds] = useState<Set<string>>(new Set());
  const previousIds = useRef<Set<string> | null>(null);
  const previousQueryKey = useRef<string | null>(null);
  const filterKey = JSON.stringify(filters);
  const queryKey = `${filterKey}:${localNodeId ?? "global"}:${localDepth}`;

  const load = useCallback(
    async (signal?: AbortSignal, quiet = false) => {
      if (!quiet) setLoading(true);
      try {
        const result = localNodeId
          ? await nexusApi.neighborhood(localNodeId, localDepth, filters, signal)
          : await nexusApi.graph(filters, signal);
        const currentIds = new Set(result.nodes.map((node) => node.id));
        if (previousIds.current && previousQueryKey.current === queryKey) {
          setNewNodeIds(new Set([...currentIds].filter((id) => !previousIds.current?.has(id))));
        } else {
          setNewNodeIds(new Set());
        }
        previousIds.current = currentIds;
        previousQueryKey.current = queryKey;
        setData((current) => (current?.snapshot_id === result.snapshot_id ? current : result));
        setError(null);
      } catch (reason) {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason : new Error("Unable to load the knowledge graph"));
      } finally {
        if (!quiet) setLoading(false);
      }
    },
    // filterKey intentionally converts the filter object into a stable primitive dependency.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [filterKey, localNodeId, localDepth, queryKey],
  );

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  useEffect(() => {
    if (pollInterval <= 0) return;
    const timer = window.setInterval(() => void load(undefined, true), pollInterval);
    return () => window.clearInterval(timer);
  }, [load, pollInterval]);

  useEffect(() => {
    if (!newNodeIds.size) return;
    const timer = window.setTimeout(() => setNewNodeIds(new Set()), 5_000);
    return () => window.clearTimeout(timer);
  }, [newNodeIds]);

  return { data, error, loading, newNodeIds, refresh: () => load() };
}

export function useNodeDetail(nodeId: string | null) {
  const [detail, setDetail] = useState<NodeDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!nodeId) {
      setDetail(null);
      setError(null);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    void nexusApi
      .node(nodeId, controller.signal)
      .then((result) => {
        setDetail(result);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason : new Error("Unable to load node details"));
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [nodeId]);

  return { detail, loading, error };
}

export function useSearch(query: string) {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const normalized = query.trim();
    if (!normalized) {
      setResults([]);
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setLoading(true);
      void nexusApi
        .search(normalized, controller.signal)
        .then(setResults)
        .catch((reason: unknown) => {
          if (!(reason instanceof DOMException && reason.name === "AbortError")) setResults([]);
        })
        .finally(() => setLoading(false));
    }, 180);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [query]);

  return { results, loading };
}

export function useOverviewData(pollInterval: number) {
  const [gaps, setGaps] = useState<Record<string, unknown>[]>([]);
  const [contradictions, setContradictions] = useState<Record<string, unknown>[]>([]);
  const [investigations, setInvestigations] = useState<InvestigationSummary[]>([]);
  const [sources, setSources] = useState<GraphNode[]>([]);
  const [runtime, setRuntime] = useState<RuntimeOverview | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      const [nextGaps, nextContradictions, nextInvestigations, nextSources, nextRuntime] = await Promise.all([
        nexusApi.gaps(signal),
        nexusApi.contradictions(signal),
        nexusApi.investigations(signal),
        nexusApi.sources(signal),
        nexusApi.runtime(signal),
      ]);
      setGaps(nextGaps);
      setContradictions(nextContradictions);
      setInvestigations(nextInvestigations);
      setSources(nextSources);
      setRuntime(nextRuntime);
      setError(null);
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === "AbortError") return;
      setError(reason instanceof Error ? reason : new Error("Unable to load NEXUS activity"));
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    if (pollInterval <= 0) return;
    const timer = window.setInterval(() => void refresh(), pollInterval);
    return () => window.clearInterval(timer);
  }, [pollInterval, refresh]);

  return { gaps, contradictions, investigations, sources, runtime, error, refresh };
}
