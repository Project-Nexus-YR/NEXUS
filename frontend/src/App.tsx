import { AlertCircle, Network, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { FilterPanel } from "./components/FilterPanel";
import { GraphCanvas, type GraphCanvasHandle } from "./components/GraphCanvas";
import { GraphControls } from "./components/GraphControls";
import { GraphLegend } from "./components/GraphLegend";
import { Inspector } from "./components/Inspector";
import { ListDrawer } from "./components/ListDrawer";
import { SearchPalette } from "./components/SearchPalette";
import { Sidebar } from "./components/Sidebar";
import { StatusBar } from "./components/StatusBar";
import { useGraphData, useNodeDetail, useOverviewData, useSearch } from "./hooks/useNexusData";
import type { GraphFilters, NodeKind, SearchResult, Selection, SidebarView } from "./types";

const ALL_KINDS: NodeKind[] = ["entity", "claim", "evidence", "document", "source", "gap", "investigation"];
const INITIAL_FILTERS: GraphFilters = {
  nodeKinds: ALL_KINDS,
  relationTypes: [],
  verificationStates: [],
  minConfidence: 0,
  maxNodes: 750,
  maxEdges: 2_500,
};

export default function App() {
  const canvas = useRef<GraphCanvasHandle>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const [activeView, setActiveView] = useState<SidebarView>("graph");
  const [selection, setSelection] = useState<Selection | null>(null);
  const [query, setQuery] = useState("");
  const [searchFocus, setSearchFocus] = useState(0);
  const [filters, setFilters] = useState<GraphFilters>(INITIAL_FILTERS);
  const [localMode, setLocalMode] = useState(false);
  const [localDepth, setLocalDepth] = useState(1);
  const [pollInterval, setPollInterval] = useState(8_000);
  const [relationTypes, setRelationTypes] = useState<string[]>([]);

  const localNodeId = localMode && selection?.type === "node" ? selection.id : null;
  const graphState = useGraphData({ filters, localNodeId, localDepth, pollInterval });
  const detailState = useNodeDetail(selection?.type === "node" ? selection.id : null);
  const searchState = useSearch(query);
  const overview = useOverviewData(pollInterval);

  useEffect(() => {
    if (!graphState.data) return;
    setRelationTypes((current) => Array.from(new Set([
      ...current,
      ...filters.relationTypes,
      ...graphState.data!.edges.map((edge) => edge.kind),
    ])).sort());
  }, [filters.relationTypes, graphState.data]);

  const selectedEdge = useMemo(
    () => selection?.type === "edge" ? graphState.data?.edges.find((edge) => edge.id === selection.id) ?? null : null,
    [graphState.data, selection],
  );
  const focusSearch = useCallback(() => {
    setSearchFocus((value) => value + 1);
    setActiveView("search");
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        focusSearch();
      }
      if (event.key === "Escape") {
        setQuery("");
        setSelection(null);
        setLocalMode(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [focusSearch]);

  useEffect(() => {
    if (selection?.type !== "node" || !graphState.data?.nodes.some((node) => node.id === selection.id)) return;
    const timer = window.setTimeout(() => canvas.current?.focusNode(selection.id), 120);
    return () => window.clearTimeout(timer);
  }, [graphState.data, selection]);

  const handleSelection = (next: Selection | null) => {
    setSelection(next);
    if (next) setInspectorCollapsed(false);
    if (!next) setLocalMode(false);
  };

  const selectNode = (nodeId: string, kind?: NodeKind, forceLocal = true) => {
    if (kind && !filters.nodeKinds.includes(kind)) {
      setFilters((current) => ({ ...current, nodeKinds: [...current.nodeKinds, kind] }));
    }
    setSelection({ type: "node", id: nodeId });
    setInspectorCollapsed(false);
    if (forceLocal) setLocalMode(true);
  };

  const selectSearchResult = (result: SearchResult) => {
    selectNode(result.id, result.kind, true);
    setQuery("");
    setActiveView("graph");
  };

  const toggleLocal = () => {
    if (selection?.type !== "node") return;
    setLocalMode((value) => !value);
  };

  const handleView = (view: SidebarView) => {
    setActiveView(view);
    if (view === "search") focusSearch();
  };

  const refreshAll = () => {
    void graphState.refresh();
    void overview.refresh();
  };

  const counts = {
    investigations: overview.investigations.length,
    gaps: overview.gaps.length,
    contradictions: overview.contradictions.length,
    sources: overview.sources.length,
    runtime: overview.runtime?.tasks.length ?? 0,
  };

  return (
    <div className="app-shell">
      <Sidebar
        collapsed={sidebarCollapsed}
        activeView={activeView}
        onToggle={() => setSidebarCollapsed((value) => !value)}
        onView={handleView}
        counts={counts}
      />
      <main className="workspace">
        <div className="workspace__canvas">
          <SearchPalette
            query={query}
            onQuery={setQuery}
            results={searchState.results}
            loading={searchState.loading}
            onSelect={selectSearchResult}
            forceFocus={searchFocus}
          />
          <GraphControls
            localMode={localMode}
            depth={localDepth}
            canFocus={selection?.type === "node"}
            onFit={() => canvas.current?.fit()}
            onZoomIn={() => canvas.current?.zoomIn()}
            onZoomOut={() => canvas.current?.zoomOut()}
            onReset={() => canvas.current?.reset()}
            onLocalMode={toggleLocal}
            onDepth={setLocalDepth}
            onRefresh={refreshAll}
          />
          <FilterPanel filters={filters} relationTypes={relationTypes} onChange={setFilters} />
          <GraphLegend />
          {graphState.loading && !graphState.data && <CanvasState icon="loading" title="Mapping knowledge" text="Building a bounded graph projection from NEXUS…" />}
          {graphState.error && !graphState.data && <CanvasState icon="error" title="Knowledge graph unavailable" text={graphState.error.message} action={() => void graphState.refresh()} />}
          {graphState.data && graphState.data.nodes.length === 0 && <CanvasState icon="empty" title="No knowledge to display" text="Load a NEXUS knowledge snapshot or adjust the active graph filters." />}
          {graphState.data && graphState.data.nodes.length > 0 && (
            <GraphCanvas
              ref={canvas}
              data={graphState.data}
              selection={selection}
              searchQuery={query}
              newNodeIds={graphState.newNodeIds}
              onSelect={handleSelection}
            />
          )}
          {graphState.data && <StatusBar graph={graphState.data} localMode={localMode} />}
          {graphState.loading && graphState.data && <div className="background-loading"><span className="mini-spinner" />Synchronizing</div>}
        </div>
        <ListDrawer
          view={activeView}
          gaps={overview.gaps}
          contradictions={overview.contradictions}
          investigations={overview.investigations}
          runtime={overview.runtime}
          sources={overview.sources}
          pollInterval={pollInterval}
          onPollInterval={setPollInterval}
          onClose={() => setActiveView("graph")}
          onSelectNode={(id) => selectNode(id)}
          onRefresh={refreshAll}
          onSearch={focusSearch}
        />
      </main>
      <Inspector
        detail={detailState.detail}
        edge={selectedEdge}
        loading={detailState.loading}
        error={detailState.error}
        collapsed={inspectorCollapsed}
        localMode={localMode}
        onCollapse={() => setInspectorCollapsed((value) => !value)}
        onClose={() => handleSelection(null)}
        onFocusNode={(id) => selectNode(id, undefined, false)}
        onLocalGraph={toggleLocal}
      />
    </div>
  );
}

function CanvasState({ icon, title, text, action }: { icon: "loading" | "error" | "empty"; title: string; text: string; action?: () => void }) {
  return <div className="canvas-state">{icon === "loading" ? <span className="spinner" /> : icon === "error" ? <AlertCircle size={26} /> : <Network size={27} />}<strong>{title}</strong><p>{text}</p>{action && <button className="primary-button" onClick={action}><RotateCcw size={13} />Retry</button>}</div>;
}
