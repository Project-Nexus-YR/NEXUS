import { Focus, LocateFixed, Maximize, Minus, Plus, RefreshCcw } from "lucide-react";

interface GraphControlsProps {
  localMode: boolean;
  depth: number;
  canFocus: boolean;
  onFit: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
  onLocalMode: () => void;
  onDepth: (depth: number) => void;
  onRefresh: () => void;
}

export function GraphControls(props: GraphControlsProps) {
  return (
    <div className="graph-controls">
      <div className="control-group">
        <button onClick={props.onZoomIn} title="Zoom in"><Plus size={15} /></button>
        <button onClick={props.onZoomOut} title="Zoom out"><Minus size={15} /></button>
        <button onClick={props.onFit} title="Zoom to fit"><Maximize size={15} /></button>
        <button onClick={props.onReset} title="Reset graph layout"><LocateFixed size={15} /></button>
      </div>
      <div className="control-group">
        <button
          className={props.localMode ? "control-active" : ""}
          onClick={props.onLocalMode}
          disabled={!props.canFocus}
          title="Show local graph"
        ><Focus size={15} /><span>Local</span></button>
        {props.localMode && [1, 2, 3].map((depth) => (
          <button key={depth} className={props.depth === depth ? "control-active" : ""} onClick={() => props.onDepth(depth)}>
            {depth}
          </button>
        ))}
      </div>
      <button className="refresh-button" onClick={props.onRefresh} title="Refresh from NEXUS"><RefreshCcw size={14} /></button>
    </div>
  );
}
