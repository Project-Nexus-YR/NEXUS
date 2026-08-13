import { CornerDownLeft, Search, X } from "lucide-react";
import { useEffect, useRef } from "react";

import type { SearchResult } from "../types";

interface SearchPaletteProps {
  query: string;
  onQuery: (value: string) => void;
  results: SearchResult[];
  loading: boolean;
  onSelect: (node: SearchResult) => void;
  forceFocus: number;
}

export function SearchPalette({ query, onQuery, results, loading, onSelect, forceFocus }: SearchPaletteProps) {
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (forceFocus) input.current?.focus();
  }, [forceFocus]);

  return (
    <div className={`search-palette ${query ? "search-palette--open" : ""}`}>
      <Search size={16} />
      <input
        ref={input}
        value={query}
        onChange={(event) => onQuery(event.target.value)}
        placeholder="Search entities, claims, sources…"
        aria-label="Search knowledge graph"
      />
      {loading && <span className="mini-spinner" />}
      {!query && <kbd>⌘ K</kbd>}
      {query && (
        <button className="bare-button" onClick={() => onQuery("")} aria-label="Clear search"><X size={14} /></button>
      )}
      {query && (
        <div className="search-results">
          {results.length === 0 && !loading && <div className="search-empty">No matching knowledge found</div>}
          {results.map((result) => (
            <button key={result.id} onClick={() => onSelect(result)}>
              <span className={`kind-dot kind-dot--${result.kind}`} />
              <span><strong>{result.label}</strong><small>{result.kind} · {result.subtitle}</small></span>
              <CornerDownLeft size={13} />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
