import {
  AlertTriangle,
  Bot,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Database,
  GitFork,
  Network,
  Search,
  Settings,
  Sparkles,
  type LucideIcon,
} from "lucide-react";

import type { SidebarView } from "../types";

interface SidebarProps {
  collapsed: boolean;
  activeView: SidebarView;
  onToggle: () => void;
  onView: (view: SidebarView) => void;
  counts: Partial<Record<SidebarView, number>>;
}

const ITEMS: Array<{ id: SidebarView; label: string; icon: LucideIcon }> = [
  { id: "graph", label: "Graph", icon: Network },
  { id: "search", label: "Search", icon: Search },
  { id: "investigations", label: "Investigations", icon: Sparkles },
  { id: "gaps", label: "Knowledge Gaps", icon: CircleHelp },
  { id: "contradictions", label: "Contradictions", icon: AlertTriangle },
  { id: "sources", label: "Sources", icon: Database },
  { id: "runtime", label: "Runtime / Agents", icon: Bot },
  { id: "settings", label: "Settings", icon: Settings },
];

export function Sidebar({ collapsed, activeView, onToggle, onView, counts }: SidebarProps) {
  return (
    <aside className={`sidebar ${collapsed ? "sidebar--collapsed" : ""}`}>
      <div className="brand">
        <div className="brand__mark"><GitFork size={19} /></div>
        {!collapsed && (
          <div className="brand__text">
            <strong>NEXUS</strong>
            <span>Knowledge runtime</span>
          </div>
        )}
      </div>
      <nav className="sidebar__nav" aria-label="Primary navigation">
        {ITEMS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`nav-item ${activeView === id ? "nav-item--active" : ""}`}
            onClick={() => onView(id)}
            title={collapsed ? label : undefined}
          >
            <Icon size={17} />
            {!collapsed && <span>{label}</span>}
            {!collapsed && counts[id] !== undefined && <em>{counts[id]}</em>}
          </button>
        ))}
      </nav>
      <div className="sidebar__footer">
        {!collapsed && <span className="connection-state"><i /> Local runtime</span>}
        <button className="icon-button" onClick={onToggle} aria-label="Toggle sidebar">
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>
    </aside>
  );
}
