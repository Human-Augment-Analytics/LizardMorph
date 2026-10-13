import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { UploadHistoryItem } from "../models/UploadHistoryItem";
import { HistoryPanel } from "./HistoryPanel";

const sampleHistory: UploadHistoryItem[] = [
  {
    name: "specimen_001.jpg",
    timestamp: "2026-10-13 14:00",
    index: 0,
    viewType: "toepads",
  },
  {
    name: "specimen_002.jpg",
    timestamp: "2026-10-13 14:05",
    index: 1,
    viewType: "toepads",
  },
];

const render = (overrides: Partial<React.ComponentProps<typeof HistoryPanel>> = {}) =>
  renderToStaticMarkup(
    <HistoryPanel
      uploadHistory={sampleHistory}
      currentImageIndex={0}
      uploadProgress={{}}
      onSelectImage={() => {}}
      onLoadFromUploads={() => {}}
      theme="light"
      {...overrides}
    />
  );

describe("HistoryPanel", () => {
  it("renders expanded state with title, table, and collapse button when not collapsed", () => {
    const onToggle = vi.fn();
    const markup = render({ isCollapsed: false, onToggleCollapse: onToggle });

    expect(markup).toContain("History");
    expect(markup).toContain("◀ Hide");
    expect(markup).toContain('aria-label="Hide history panel"');
    expect(markup).toContain('title="Hide history panel"');
    expect(markup).toContain("specimen_001.jpg");
    expect(markup).toContain("specimen_002.jpg");
    expect(markup).not.toContain("data-testid=\"history-panel-collapsed\"");
  });

  it("omits the collapse button if onToggleCollapse is not provided", () => {
    const markup = render({ isCollapsed: false });

    expect(markup).toContain("History");
    expect(markup).not.toContain("◀ Hide");
  });

  it("renders a slim collapsed bar when isCollapsed is true", () => {
    const onToggle = vi.fn();
    const markup = render({ isCollapsed: true, onToggleCollapse: onToggle });

    expect(markup).toContain("data-testid=\"history-panel-collapsed\"");
    expect(markup).toContain("width:36px");
    expect(markup).toContain("▶");
    expect(markup).toContain("📂 History");
    expect(markup).toContain('aria-label="Show history panel"');
    expect(markup).toContain('title="Show history"');
    // History table items should not be visible in collapsed mode
    expect(markup).not.toContain("specimen_001.jpg");
    expect(markup).not.toContain("<table");
  });

  it("renders correctly in dark mode", () => {
    const markupCollapsed = render({ isCollapsed: true, theme: "dark" });
    expect(markupCollapsed).toContain("▶");
    expect(markupCollapsed).toContain("📂 History");

    const markupExpanded = render({ isCollapsed: false, onToggleCollapse: () => {}, theme: "dark" });
    expect(markupExpanded).toContain("◀ Hide");
    expect(markupExpanded).toContain("specimen_001.jpg");
  });

  it("includes aria-expanded=true in expanded mode and aria-expanded=false in collapsed mode", () => {
    const markupExpanded = render({ isCollapsed: false, onToggleCollapse: () => {} });
    expect(markupExpanded).toContain('aria-expanded="true"');

    const markupCollapsed = render({ isCollapsed: true, onToggleCollapse: () => {} });
    expect(markupCollapsed).toContain('aria-expanded="false"');
  });

  it("renders history table rows with keyboard accessibility attributes", () => {
    const markup = render({ isCollapsed: false });
    expect(markup).toContain('tabindex="0"');
    expect(markup).toContain('role="button"');
    expect(markup).toContain('aria-label="Select specimen_001.jpg"');
    expect(markup).toContain('aria-selected="true"');
  });

  it("handles Enter and Space keydown events to toggle collapse in collapsed mode", () => {
    const onToggle = vi.fn();
    const panel = new HistoryPanel({
      uploadHistory: sampleHistory,
      currentImageIndex: 0,
      uploadProgress: {},
      onSelectImage: () => {},
      onLoadFromUploads: () => {},
      isCollapsed: true,
      onToggleCollapse: onToggle,
      theme: "light",
    });

    const rendered = panel.render() as React.ReactElement<{
      children: React.ReactElement<{
        onKeyDown: (e: { key: string; preventDefault: () => void; stopPropagation: () => void }) => void;
      }>;
    }>;

    const preventDefault = vi.fn();
    const stopPropagation = vi.fn();

    const button = rendered.props.children;
    button.props.onKeyDown({ key: "Enter", preventDefault, stopPropagation });
    expect(onToggle).toHaveBeenCalledTimes(1);
    expect(preventDefault).toHaveBeenCalledTimes(1);

    button.props.onKeyDown({ key: " ", preventDefault, stopPropagation });
    expect(onToggle).toHaveBeenCalledTimes(2);
  });

  it("handles Enter and Space keydown on history row to select an image", () => {
    const onSelect = vi.fn();
    const panel = new HistoryPanel({
      uploadHistory: sampleHistory,
      currentImageIndex: 0,
      uploadProgress: {},
      onSelectImage: onSelect,
      onLoadFromUploads: () => {},
      isCollapsed: false,
      theme: "light",
    });

    const rendered = panel.render() as React.ReactElement<{
      children: [
        React.ReactElement,
        React.ReactElement<{
          children: React.ReactElement<{
            children: [
              React.ReactElement,
              React.ReactElement<{
                children: Array<React.ReactElement<{
                  onKeyDown: (e: { key: string; preventDefault: () => void }) => void;
                }>>;
              }>
            ];
          }>;
        }>
      ];
    }>;

    const tableContainer = rendered.props.children[1];
    const table = tableContainer.props.children;
    const tbody = table.props.children[1];
    const firstRow = tbody.props.children[0];

    const preventDefault = vi.fn();
    firstRow.props.onKeyDown({ key: "Enter", preventDefault });
    expect(onSelect).toHaveBeenCalledWith(0);
    expect(preventDefault).toHaveBeenCalled();
  });

  it("renders empty state message when upload history is empty", () => {
    const markup = render({ uploadHistory: [] });

    expect(markup).toContain("No images in history");
  });
});
