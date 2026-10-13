import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import * as d3 from "d3";
import { SVGViewer } from "./SVGViewer";
import type { Point } from "../models/Point";

const sharedPoints: Point[] = [{ id: 0, x: 50, y: 50 }];

const createDefaultProps = (
  overrides: Partial<React.ComponentProps<typeof SVGViewer>> = {}
) => ({
  selectedViewType: "toepads",
  dataFetched: true,
  loading: false,
  dataLoading: false,
  dataError: null,
  uploadHistory: [],
  scatterData: sharedPoints,
  originalScatterData: sharedPoints,
  selectedPoint: null,
  needsScaling: false,
  currentImageURL: "/test.jpg",
  imageWidth: 800,
  imageHeight: 600,
  zoomTransform: d3.zoomIdentity,
  onPointSelect: vi.fn(),
  onScatterDataUpdate: vi.fn(),
  onScalingComplete: vi.fn(),
  onZoomChange: vi.fn(),
  isEditMode: false,
  onToggleEditMode: vi.fn(),
  onResetZoom: vi.fn(),
  theme: "light" as const,
  ...overrides,
});

describe("SVGViewer edit mode zoom/pan hint", () => {
  it("does not render the edit mode hint in view mode", () => {
    const markup = renderToStaticMarkup(
      <SVGViewer {...createDefaultProps({ isEditMode: false })} />
    );

    expect(markup).not.toContain(
      "💡 Hold Ctrl + Scroll to zoom, Ctrl + Drag to pan in Edit Mode"
    );
  });

  it("renders the sleek hint overlay in edit mode", () => {
    const markup = renderToStaticMarkup(
      <SVGViewer {...createDefaultProps({ isEditMode: true })} />
    );

    expect(markup).toContain(
      "💡 Hold Ctrl + Scroll to zoom, Ctrl + Drag to pan in Edit Mode"
    );
    expect(markup).toContain('data-testid="edit-mode-zoom-hint"');
  });

  it("includes Ctrl + Scroll / Drag shortcut in the controls panel markup", () => {
    const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
    Object.assign(viewer.state, { showControls: true });
    const markup = renderToStaticMarkup(viewer.render());

    expect(markup).toContain("Ctrl + Scroll / Drag");
  });

  it("dismisses hint when dismissHint is called", () => {
    const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
    expect(viewer.state.hintDismissed).toBe(false);

    viewer.setState = vi.fn().mockImplementation((updater) => {
      const next =
        typeof updater === "function"
          ? updater(viewer.state, viewer.props)
          : updater;
      Object.assign(viewer.state, next);
    });

    viewer.dismissHint();
    expect(viewer.state.hintDismissed).toBe(true);

    const rendered = viewer.render();
    const markup = renderToStaticMarkup(rendered);
    expect(markup).not.toContain('data-testid="edit-mode-zoom-hint"');
  });

  it("adapts hint overlay and controls styling according to light and dark theme", () => {
    const lightMarkup = renderToStaticMarkup(
      <SVGViewer {...createDefaultProps({ isEditMode: true, theme: "light" })} />
    );
    expect(lightMarkup).toContain("rgba(255, 255, 255, 0.94)");

    const darkMarkup = renderToStaticMarkup(
      <SVGViewer {...createDefaultProps({ isEditMode: true, theme: "dark" })} />
    );
    expect(darkMarkup).toContain("rgba(20, 24, 33, 0.90)");
  });

  it("suppresses hint overlay when data is not fetched, loading, or in error state", () => {
    const notFetchedMarkup = renderToStaticMarkup(
      <SVGViewer {...createDefaultProps({ isEditMode: true, dataFetched: false })} />
    );
    expect(notFetchedMarkup).not.toContain('data-testid="edit-mode-zoom-hint"');

    const loadingMarkup = renderToStaticMarkup(
      <SVGViewer {...createDefaultProps({ isEditMode: true, dataLoading: true })} />
    );
    expect(loadingMarkup).not.toContain('data-testid="edit-mode-zoom-hint"');

    const errorMarkup = renderToStaticMarkup(
      <SVGViewer {...createDefaultProps({ isEditMode: true, dataError: new Error("Failed") })} />
    );
    expect(errorMarkup).not.toContain('data-testid="edit-mode-zoom-hint"');
  });

  it("applies grab cursor in view mode and crosshair in edit mode", () => {
    const viewMarkup = renderToStaticMarkup(
      <SVGViewer {...createDefaultProps({ isEditMode: false })} />
    );
    expect(viewMarkup).toContain("cursor:grab");

    const editMarkup = renderToStaticMarkup(
      <SVGViewer {...createDefaultProps({ isEditMode: true })} />
    );
    expect(editMarkup).toContain("cursor:crosshair");
  });
});

describe("SVGViewer d3 zoom filter", () => {
  describe("in View Mode (isEditMode: false)", () => {
    it("allows standard wheel zoom without Ctrl", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: false }));
      const wheelEvent = {
        type: "wheel",
        button: 0,
        ctrlKey: false,
        metaKey: false,
      } as unknown as Event;

      expect(viewer.zoomFilter(wheelEvent)).toBe(true);
    });

    it("allows standard mousedown pan without Ctrl", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: false }));
      const mouseEvent = {
        type: "mousedown",
        button: 0,
        ctrlKey: false,
        metaKey: false,
      } as unknown as Event;

      expect(viewer.zoomFilter(mouseEvent)).toBe(true);
    });

    it("blocks double click zoom", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: false }));
      const preventDefault = vi.fn();
      const dblClickEvent = {
        type: "dblclick",
        button: 0,
        preventDefault,
      } as unknown as Event;

      expect(viewer.zoomFilter(dblClickEvent)).toBe(false);
      expect(preventDefault).toHaveBeenCalled();
    });

    it("blocks non-primary buttons (e.g. right click for context menu)", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: false }));
      const rightClickEvent = {
        type: "mousedown",
        button: 2,
        ctrlKey: false,
      } as unknown as Event;

      expect(viewer.zoomFilter(rightClickEvent)).toBe(false);
    });
  });

  describe("in Edit Mode (isEditMode: true)", () => {
    it("blocks wheel zoom when Ctrl and Meta are NOT pressed", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const wheelEvent = {
        type: "wheel",
        button: 0,
        ctrlKey: false,
        metaKey: false,
      } as unknown as Event;

      expect(viewer.zoomFilter(wheelEvent)).toBe(false);
    });

    it("blocks canvas drag/pan when Ctrl and Meta are NOT pressed", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const mouseEvent = {
        type: "mousedown",
        button: 0,
        ctrlKey: false,
        metaKey: false,
      } as unknown as Event;

      expect(viewer.zoomFilter(mouseEvent)).toBe(false);
    });

    it("allows wheel zoom when Ctrl is pressed", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const wheelEvent = {
        type: "wheel",
        button: 0,
        ctrlKey: true,
        metaKey: false,
      } as unknown as Event;

      expect(viewer.zoomFilter(wheelEvent)).toBe(true);
    });

    it("allows canvas pan when Ctrl is pressed", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const mouseEvent = {
        type: "mousedown",
        button: 0,
        ctrlKey: true,
        metaKey: false,
      } as unknown as Event;

      expect(viewer.zoomFilter(mouseEvent)).toBe(true);
    });

    it("allows wheel zoom when Meta (Mac Cmd) is pressed", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const wheelEvent = {
        type: "wheel",
        button: 0,
        ctrlKey: false,
        metaKey: true,
      } as unknown as Event;

      expect(viewer.zoomFilter(wheelEvent)).toBe(true);
    });

    it("allows canvas pan when Meta (Mac Cmd) is pressed", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const mouseEvent = {
        type: "mousedown",
        button: 0,
        ctrlKey: false,
        metaKey: true,
      } as unknown as Event;

      expect(viewer.zoomFilter(mouseEvent)).toBe(true);
    });

    it("blocks double click even when Ctrl is pressed", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const preventDefault = vi.fn();
      const dblClickEvent = {
        type: "dblclick",
        button: 0,
        ctrlKey: true,
        preventDefault,
      } as unknown as Event;

      expect(viewer.zoomFilter(dblClickEvent)).toBe(false);
      expect(preventDefault).toHaveBeenCalled();
    });

    it("blocks right click even when Ctrl is pressed", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const rightClickEvent = {
        type: "mousedown",
        button: 2,
        ctrlKey: true,
      } as unknown as Event;

      expect(viewer.zoomFilter(rightClickEvent)).toBe(false);
    });
  });
});

describe("SVGViewer landmark drag interaction (dragstarted)", () => {
  it("does not initiate landmark drag when Ctrl is pressed so event passes to canvas panning", () => {
    const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
    const stopPropagation = vi.fn();
    const sourceEvent = {
      ctrlKey: true,
      metaKey: false,
      stopPropagation,
    } as unknown as MouseEvent;

    const dragEvent = {
      sourceEvent,
    } as unknown as d3.D3DragEvent<SVGGElement, Point, Point>;

    const landmark: Point = { id: 0, x: 50, y: 50 };

    viewer.dragstarted(dragEvent, landmark);

    // stopPropagation should NOT have been called, letting canvas handle panning
    expect(stopPropagation).not.toHaveBeenCalled();
    // drag state should remain inactive
    expect(viewer["isDragging"]).toBe(false);
    expect(viewer["draggedLandmarkId"]).toBeNull();
  });

  it("does not initiate landmark drag when Meta is pressed so event passes to canvas panning", () => {
    const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
    const stopPropagation = vi.fn();
    const sourceEvent = {
      ctrlKey: false,
      metaKey: true,
      stopPropagation,
    } as unknown as MouseEvent;

    const dragEvent = {
      sourceEvent,
    } as unknown as d3.D3DragEvent<SVGGElement, Point, Point>;

    const landmark: Point = { id: 0, x: 50, y: 50 };

    viewer.dragstarted(dragEvent, landmark);

    expect(stopPropagation).not.toHaveBeenCalled();
    expect(viewer["isDragging"]).toBe(false);
  });

  it("initiates landmark dragging when neither Ctrl nor Meta is pressed", () => {
    const onPointSelect = vi.fn();
    const viewer = new SVGViewer(
      createDefaultProps({
        isEditMode: true,
        selectedPoint: { id: 0, x: 50, y: 50 },
        onPointSelect,
      })
    );

    const stopPropagation = vi.fn();
    const mockParent = {
      appendChild: vi.fn(),
    };
    const mockGroup = {
      parentNode: mockParent,
    };
    const mockTarget = {
      parentNode: mockGroup,
    };

    const sourceEvent = {
      ctrlKey: false,
      metaKey: false,
      stopPropagation,
      target: mockTarget,
    } as unknown as MouseEvent;

    const dragEvent = {
      sourceEvent,
    } as unknown as d3.D3DragEvent<SVGGElement, Point, Point>;

    const landmark: Point = { id: 0, x: 50, y: 50 };

    viewer.dragstarted(dragEvent, landmark);

    expect(stopPropagation).toHaveBeenCalled();
    expect(viewer["isDragging"]).toBe(true);
    expect(viewer["draggedLandmarkId"]).toBe(0);
  });

  it("dragended safely ignores completion if isDragging is false", () => {
    const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
    const stopPropagation = vi.fn();
    const sourceEvent = {
      stopPropagation,
    } as unknown as MouseEvent;

    const dragEvent = {
      sourceEvent,
    } as unknown as d3.D3DragEvent<SVGGElement, Point, Point>;

    viewer.dragended(dragEvent);
    expect(stopPropagation).not.toHaveBeenCalled();
  });
});

describe("SVGViewer zoom transform synchronization", () => {
  it("syncs zoom transform on componentDidUpdate when switching to edit mode", () => {
    const prevProps = createDefaultProps({ isEditMode: false });
    const nextTransform = d3.zoomIdentity.translate(100, 50).scale(2);
    const nextProps = createDefaultProps({
      isEditMode: true,
      zoomTransform: nextTransform,
    });

    const viewer = new SVGViewer(nextProps);
    const mockSvgElement = {
      __zoom: d3.zoomIdentity,
      querySelector: vi.fn().mockReturnValue(null),
      querySelectorAll: vi.fn().mockReturnValue([]),
      getAttribute: vi.fn().mockReturnValue("100"),
      ownerDocument: null,
      namespaceURI: "http://www.w3.org/2000/svg",
    } as unknown as SVGSVGElement;
    const mockZoom = d3.zoom<SVGSVGElement, unknown>();
    const zoomTransformSpy = vi.fn();

    mockZoom.transform = zoomTransformSpy as unknown as typeof mockZoom.transform;
    (viewer.svgRef as unknown as { current: SVGSVGElement }).current = mockSvgElement;
    (viewer.zoomRef as unknown as { current: d3.ZoomBehavior<SVGSVGElement, unknown> }).current = mockZoom;

    viewer.componentDidUpdate(prevProps);

    expect(zoomTransformSpy).toHaveBeenCalled();
  });

  it("syncs zoom transform on componentDidUpdate when zoomTransform prop changes in edit mode", () => {
    const prevProps = createDefaultProps({
      isEditMode: true,
      zoomTransform: d3.zoomIdentity,
    });
    const nextTransform = d3.zoomIdentity.translate(20, 30).scale(1.5);
    const nextProps = createDefaultProps({
      isEditMode: true,
      zoomTransform: nextTransform,
    });

    const viewer = new SVGViewer(nextProps);
    const mockSvgElement = {
      __zoom: d3.zoomIdentity,
      querySelector: vi.fn().mockReturnValue(null),
      querySelectorAll: vi.fn().mockReturnValue([]),
      getAttribute: vi.fn().mockReturnValue("100"),
      ownerDocument: null,
      namespaceURI: "http://www.w3.org/2000/svg",
    } as unknown as SVGSVGElement;
    const mockZoom = d3.zoom<SVGSVGElement, unknown>();
    const zoomTransformSpy = vi.fn();

    mockZoom.transform = zoomTransformSpy as unknown as typeof mockZoom.transform;
    (viewer.svgRef as unknown as { current: SVGSVGElement }).current = mockSvgElement;
    (viewer.zoomRef as unknown as { current: d3.ZoomBehavior<SVGSVGElement, unknown> }).current = mockZoom;

    viewer.componentDidUpdate(prevProps);

    expect(zoomTransformSpy).toHaveBeenCalled();
  });
});
