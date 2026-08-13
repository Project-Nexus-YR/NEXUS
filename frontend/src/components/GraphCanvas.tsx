import forceAtlas2 from "graphology-layout-forceatlas2";
import FA2Layout from "graphology-layout-forceatlas2/worker";
import type { Attributes } from "graphology-types";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import Sigma from "sigma";

import {
  buildGraph,
  edgeColor,
  nodeColor,
  type VisualEdgeAttributes,
  type VisualNodeAttributes,
} from "../graph/adapter";
import type { GraphResponse, Selection } from "../types";

export interface GraphCanvasHandle {
  fit: () => void;
  zoomIn: () => void;
  zoomOut: () => void;
  reset: () => void;
  focusNode: (nodeId: string) => void;
}

interface GraphCanvasProps {
  data: GraphResponse;
  selection: Selection | null;
  searchQuery: string;
  newNodeIds: Set<string>;
  onSelect: (selection: Selection | null) => void;
}

interface Tooltip {
  x: number;
  y: number;
  title: string;
  subtitle: string;
  kind: string;
}

export const GraphCanvas = forwardRef<GraphCanvasHandle, GraphCanvasProps>(function GraphCanvas(
  { data, selection, searchQuery, newNodeIds, onSelect },
  ref,
) {
  const container = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<Sigma<VisualNodeAttributes, VisualEdgeAttributes, Attributes> | null>(null);
  const graphRef = useRef<ReturnType<typeof buildGraph> | null>(null);
  const selectionRef = useRef(selection);
  const searchRef = useRef(searchQuery);
  const newNodesRef = useRef(newNodeIds);
  const onSelectRef = useRef(onSelect);
  const [tooltip, setTooltip] = useState<Tooltip | null>(null);

  selectionRef.current = selection;
  searchRef.current = searchQuery;
  newNodesRef.current = newNodeIds;
  onSelectRef.current = onSelect;

  useImperativeHandle(ref, () => ({
    fit: () => rendererRef.current?.getCamera().animatedReset({ duration: 260 }),
    zoomIn: () => {
      const camera = rendererRef.current?.getCamera();
      if (camera) void camera.animate({ ratio: camera.ratio / 1.45 }, { duration: 180 });
    },
    zoomOut: () => {
      const camera = rendererRef.current?.getCamera();
      if (camera) void camera.animate({ ratio: camera.ratio * 1.45 }, { duration: 180 });
    },
    reset: () => rendererRef.current?.getCamera().animatedReset({ duration: 280 }),
    focusNode: (nodeId) => {
      const renderer = rendererRef.current;
      const position = renderer?.getNodeDisplayData(nodeId);
      if (renderer && position) {
        void renderer.getCamera().animate({ x: position.x, y: position.y, ratio: 0.3 }, { duration: 420 });
      }
    },
  }), []);

  useEffect(() => {
    if (!container.current) return;
    const graph = buildGraph(data);
    graphRef.current = graph;
    const renderer = new Sigma(graph, container.current, {
      allowInvalidContainer: true,
      defaultEdgeType: "line",
      enableEdgeEvents: true,
      labelColor: { color: "#c9d0d9" },
      labelDensity: 0.22,
      labelGridCellSize: 175,
      labelRenderedSizeThreshold: 9.5,
      labelSize: 9,
      renderEdgeLabels: true,
      edgeLabelColor: { color: "#75808d" },
      edgeLabelSize: 10,
      stagePadding: 42,
      minCameraRatio: 0.05,
      maxCameraRatio: 6,
      nodeReducer: (nodeId, attributes) => {
        const raw = attributes.raw;
        const selected = selectionRef.current?.type === "node" && selectionRef.current.id === nodeId;
        const query = searchRef.current.trim().toLocaleLowerCase();
        const matches = !query || `${raw.label} ${raw.subtitle} ${raw.kind}`.toLocaleLowerCase().includes(query);
        const isNew = newNodesRef.current.has(nodeId);
        return {
          ...attributes,
          color: !matches ? "#252b31" : isNew ? "#eefbf6" : nodeColor(raw),
          size: attributes.size * (selected ? 1.65 : isNew ? 1.35 : 1),
          highlighted: selected || isNew,
          forceLabel: selected || isNew || attributes.forceLabel,
          label: matches ? attributes.label : "",
          zIndex: selected || isNew ? 4 : 1,
        };
      },
      edgeReducer: (edgeId, attributes) => {
        const raw = attributes.raw;
        const selected = selectionRef.current?.type === "edge" && selectionRef.current.id === edgeId;
        const query = searchRef.current.trim();
        return {
          ...attributes,
          color: query ? "#252b31" : selected ? "#f0d08a" : edgeColor(raw),
          size: attributes.size * (selected ? 2.2 : 1),
          label: selected ? attributes.label : "",
          zIndex: selected ? 3 : 0,
        };
      },
    });
    rendererRef.current = renderer;

    renderer.on("clickNode", ({ node }) => onSelectRef.current({ type: "node", id: node }));
    renderer.on("clickEdge", ({ edge }) => onSelectRef.current({ type: "edge", id: edge }));
    renderer.on("clickStage", () => onSelectRef.current(null));
    renderer.on("enterNode", ({ node }) => {
      const attributes = graph.getNodeAttributes(node);
      const position = renderer.graphToViewport(attributes);
      setTooltip({
        x: position.x,
        y: position.y,
        title: attributes.raw.label,
        subtitle: attributes.raw.subtitle,
        kind: attributes.raw.kind,
      });
      container.current?.classList.add("graph-canvas--hovering");
    });
    renderer.on("leaveNode", () => {
      setTooltip(null);
      container.current?.classList.remove("graph-canvas--hovering");
    });

    let draggedNode: string | null = null;
    let dragging = false;
    renderer.on("downNode", ({ node }) => {
      draggedNode = node;
      dragging = true;
      graph.setNodeAttribute(node, "highlighted", true);
      if (!renderer.getCustomBBox()) renderer.setCustomBBox(renderer.getBBox());
    });
    const mouse = renderer.getMouseCaptor();
    mouse.on("mousemovebody", (event) => {
      if (!dragging || !draggedNode) return;
      const position = renderer.viewportToGraph(event);
      graph.setNodeAttribute(draggedNode, "x", position.x);
      graph.setNodeAttribute(draggedNode, "y", position.y);
      event.preventSigmaDefault();
      event.original.preventDefault();
      event.original.stopPropagation();
    });
    mouse.on("mouseup", () => {
      if (draggedNode) graph.removeNodeAttribute(draggedNode, "highlighted");
      dragging = false;
      draggedNode = null;
    });

    let layout: FA2Layout | null = null;
    let layoutTimer = 0;
    if (graph.order > 1) {
      layout = new FA2Layout(graph, { settings: forceAtlas2.inferSettings(graph) });
      layout.start();
      layoutTimer = window.setTimeout(() => layout?.stop(), Math.min(2_400, 700 + graph.order * 3));
    }
    const fitTimer = window.setTimeout(() => renderer.getCamera().animatedReset({ duration: 320 }), 260);

    return () => {
      window.clearTimeout(layoutTimer);
      window.clearTimeout(fitTimer);
      layout?.kill();
      renderer.kill();
      rendererRef.current = null;
      graphRef.current = null;
    };
  }, [data]);

  useEffect(() => {
    rendererRef.current?.refresh();
  }, [selection, searchQuery, newNodeIds]);

  return (
    <div className="graph-canvas-wrap">
      <div ref={container} className="graph-canvas" aria-label="Interactive NEXUS knowledge graph" />
      {tooltip && (
        <div className="graph-tooltip" style={{ left: tooltip.x + 16, top: tooltip.y - 10 }}>
          <span>{tooltip.kind}</span><strong>{tooltip.title}</strong><small>{tooltip.subtitle}</small>
        </div>
      )}
    </div>
  );
});
