import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { GraphFilters } from "../types";
import { FilterPanel } from "./FilterPanel";

const filters: GraphFilters = {
  nodeKinds: ["entity", "claim", "evidence", "document", "source", "gap", "investigation"],
  relationTypes: [],
  verificationStates: [],
  minConfidence: 0,
  maxNodes: 750,
  maxEdges: 2_500,
};

describe("FilterPanel", () => {
  it("emits semantic node and edge filter changes", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <FilterPanel
        filters={filters}
        relationTypes={["contradiction", "works_at"]}
        onChange={onChange}
      />,
    );

    await user.click(screen.getByText("Claims"));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      nodeKinds: expect.not.arrayContaining(["claim"]),
    }));

    await user.click(screen.getByText("works at"));
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({
      relationTypes: ["works_at"],
    }));
  });

  it("supports contradiction and provenance quick filters", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <FilterPanel filters={filters} relationTypes={["contradiction"]} onChange={onChange} />,
    );

    await user.click(screen.getByRole("button", { name: "Contradictions only" }));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      nodeKinds: ["claim"],
      relationTypes: ["contradiction"],
    }));

    await user.click(screen.getByRole("button", { name: "Hide provenance nodes" }));
    const last = onChange.mock.calls.at(-1)?.[0] as GraphFilters;
    expect(last.nodeKinds).not.toEqual(expect.arrayContaining(["source", "document", "evidence"]));
  });
});
