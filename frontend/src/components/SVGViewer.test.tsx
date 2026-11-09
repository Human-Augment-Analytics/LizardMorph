import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import * as d3 from "d3";
import { SVGViewer, computeZoomTransformForCrop } from "./SVGViewer";
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
      "💡 Hold Shift (or Alt) + Scroll to zoom, Shift + Drag to pan in Edit Mode"
    );
  });

  it("renders the sleek hint overlay in edit mode", () => {
    const markup = renderToStaticMarkup(
      <SVGViewer {...createDefaultProps({ isEditMode: true })} />
    );

    expect(markup).toContain(
      "💡 Hold Shift (or Alt) + Scroll to zoom, Shift + Drag to pan in Edit Mode"
    );
    expect(markup).toContain('data-testid="edit-mode-zoom-hint"');
  });

  it("includes Shift / Alt + Scroll / Drag shortcut in the controls panel markup", () => {
    const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
    Object.assign(viewer.state, { showControls: true });
    const markup = renderToStaticMarkup(viewer.render());

    expect(markup).toContain("Shift / Alt + Scroll / Drag");
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
    it("allows standard wheel zoom without Shift/Alt and prevents default", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: false }));
      const preventDefault = vi.fn();
      const wheelEvent = {
        type: "wheel",
        button: 0,
        shiftKey: false,
        altKey: false,
        preventDefault,
      } as unknown as Event;

      expect(viewer.zoomFilter(wheelEvent)).toBe(true);
      expect(preventDefault).toHaveBeenCalled();
    });

    it("allows standard mousedown pan without Shift/Alt", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: false }));
      const mouseEvent = {
        type: "mousedown",
        button: 0,
        shiftKey: false,
        altKey: false,
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
        shiftKey: false,
      } as unknown as Event;

      expect(viewer.zoomFilter(rightClickEvent)).toBe(false);
    });
  });

  describe("in Edit Mode (isEditMode: true)", () => {
    it("blocks wheel zoom when Shift and Alt are NOT pressed but prevents default", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const preventDefault = vi.fn();
      const wheelEvent = {
        type: "wheel",
        button: 0,
        shiftKey: false,
        altKey: false,
        preventDefault,
      } as unknown as Event;

      expect(viewer.zoomFilter(wheelEvent)).toBe(false);
      expect(preventDefault).toHaveBeenCalled();
    });

    it("blocks canvas drag/pan when Shift and Alt are NOT pressed", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const mouseEvent = {
        type: "mousedown",
        button: 0,
        shiftKey: false,
        altKey: false,
      } as unknown as Event;

      expect(viewer.zoomFilter(mouseEvent)).toBe(false);
    });

    it("allows wheel zoom when Shift is pressed and prevents default", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const preventDefault = vi.fn();
      const wheelEvent = {
        type: "wheel",
        button: 0,
        shiftKey: true,
        altKey: false,
        preventDefault,
      } as unknown as Event;

      expect(viewer.zoomFilter(wheelEvent)).toBe(true);
      expect(preventDefault).toHaveBeenCalled();
    });

    it("allows canvas pan when Shift is pressed", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const mouseEvent = {
        type: "mousedown",
        button: 0,
        shiftKey: true,
        altKey: false,
      } as unknown as Event;

      expect(viewer.zoomFilter(mouseEvent)).toBe(true);
    });

    it("allows wheel zoom when Alt is pressed and prevents default", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const preventDefault = vi.fn();
      const wheelEvent = {
        type: "wheel",
        button: 0,
        shiftKey: false,
        altKey: true,
        preventDefault,
      } as unknown as Event;

      expect(viewer.zoomFilter(wheelEvent)).toBe(true);
      expect(preventDefault).toHaveBeenCalled();
    });

    it("allows canvas pan when Alt is pressed", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const mouseEvent = {
        type: "mousedown",
        button: 0,
        shiftKey: false,
        altKey: true,
      } as unknown as Event;

      expect(viewer.zoomFilter(mouseEvent)).toBe(true);
    });

    it("blocks double click even when Shift is pressed", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const preventDefault = vi.fn();
      const dblClickEvent = {
        type: "dblclick",
        button: 0,
        shiftKey: true,
        preventDefault,
      } as unknown as Event;

      expect(viewer.zoomFilter(dblClickEvent)).toBe(false);
      expect(preventDefault).toHaveBeenCalled();
    });

    it("blocks right click even when Shift is pressed", () => {
      const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
      const rightClickEvent = {
        type: "mousedown",
        button: 2,
        shiftKey: true,
      } as unknown as Event;

      expect(viewer.zoomFilter(rightClickEvent)).toBe(false);
    });
  });
});

describe("SVGViewer landmark drag interaction (dragstarted)", () => {
  it("does not initiate landmark drag when Shift is pressed so event passes to canvas panning", () => {
    const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
    const stopPropagation = vi.fn();
    const sourceEvent = {
      shiftKey: true,
      altKey: false,
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

  it("does not initiate landmark drag when Alt is pressed so event passes to canvas panning", () => {
    const viewer = new SVGViewer(createDefaultProps({ isEditMode: true }));
    const stopPropagation = vi.fn();
    const sourceEvent = {
      shiftKey: false,
      altKey: true,
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

  it("initiates landmark dragging when neither Shift nor Alt is pressed", () => {
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
      shiftKey: false,
      altKey: false,
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

describe("computeZoomTransformForCrop", () => {
  it("calculates correct scale and translation to center crop in viewport", () => {
    const crop = { x: 450, y: 250, size: 100 };
    const transform = computeZoomTransformForCrop(crop, 2000, 1000, 800, 400);

    // scaleRatio = 800 / 2000 = 0.4
    // cropSizeSvg = 100 * 0.4 = 40
    // targetK = (400 * 0.75) / 40 = 7.5
    expect(transform.k).toBe(7.5);

    // Center in unzoomed SVG: (450 + 50) * 0.4 = 200, (250 + 50) * 0.4 = 120
    // tx = 800 / 2 - 7.5 * 200 = 400 - 1500 = -1100
    // ty = 400 / 2 - 7.5 * 120 = 200 - 900 = -700
    expect(transform.x).toBe(-1100);
    expect(transform.y).toBe(-700);

    // Transforming the crop center point should result in viewport center (400, 200)
    const [screenX, screenY] = transform.apply([200, 120]);
    expect(screenX).toBe(400);
    expect(screenY).toBe(200);
  });

  it("returns zoomIdentity if inputs are invalid or zero", () => {
    expect(computeZoomTransformForCrop({ x: 0, y: 0, size: 0 }, 100, 100, 100, 100)).toEqual(d3.zoomIdentity);
    expect(computeZoomTransformForCrop({ x: 0, y: 0, size: 10 }, 0, 100, 100, 100)).toEqual(d3.zoomIdentity);
  });

  it("clamps zoom scale factor k to 18 when crop is very small", () => {
    // Very tiny crop in a high-res image
    const crop = { x: 100, y: 100, size: 5 };
    const transform = computeZoomTransformForCrop(crop, 4000, 3000, 800, 600);
    expect(transform.k).toBe(18.0);
  });

  it("clamps zoom scale factor k to minimum 1.0 when crop is very large", () => {
    // Crop larger than the image/viewport
    const crop = { x: 0, y: 0, size: 2500 };
    const transform = computeZoomTransformForCrop(crop, 2000, 1000, 800, 400);
    expect(transform.k).toBe(1.0);
  });

  it("handles portrait/tall viewports by scaling to the narrower width dimension", () => {
    // Viewport with width: 400, height: 800 (smaller dimension is 400)
    const crop = { x: 200, y: 200, size: 100 };
    const transform = computeZoomTransformForCrop(crop, 1000, 2000, 400, 800);
    // scaleRatio = 400 / 1000 = 0.4
    // cropSizeSvg = 100 * 0.4 = 40
    // targetK = (min(400, 800) * 0.75) / 40 = (300) / 40 = 7.5
    expect(transform.k).toBe(7.5);
  });
});

describe("SVGViewer zoomToImageRegion", () => {
  it("computes and applies zoom transform to D3 zoom and fires onZoomChange", () => {
    const onZoomChange = vi.fn();
    const props = createDefaultProps({
      imageWidth: 2000,
      imageHeight: 1000,
      onZoomChange,
    });
    const viewer = new SVGViewer(props);
    const mockZoom = d3.zoom<SVGSVGElement, unknown>();
    const zoomTransformSpy = vi.fn();
    mockZoom.transform = zoomTransformSpy as unknown as typeof mockZoom.transform;
    (viewer.zoomRef as unknown as { current: d3.ZoomBehavior<SVGSVGElement, unknown> }).current = mockZoom;

    const mockSvg = {
      getAttribute: vi.fn((attr: string) => (attr === "width" ? "800" : attr === "height" ? "400" : null)),
      node: () => ({}),
    } as unknown as SVGSVGElement;
    (viewer.svgRef as unknown as { current: SVGSVGElement }).current = mockSvg;

    const transform = viewer.zoomToImageRegion({ x: 450, y: 250, size: 100 });
    expect(transform).not.toBeNull();
    expect(transform?.k).toBe(7.5);
    expect(zoomTransformSpy).toHaveBeenCalled();
    expect(onZoomChange).toHaveBeenCalledWith(transform);
  });

  it("returns null from zoomToImageRegion when svgRef or zoomRef is not attached", () => {
    const viewer = new SVGViewer(createDefaultProps());
    expect(viewer.zoomToImageRegion({ x: 10, y: 10, size: 50 })).toBeNull();
  });

  it("prefers renderedWidth and renderedHeight if set on the viewer instance", () => {
    const onZoomChange = vi.fn();
    const props = createDefaultProps({
      imageWidth: 2000,
      imageHeight: 1000,
      onZoomChange,
    });
    const viewer = new SVGViewer(props);
    const mockZoom = d3.zoom<SVGSVGElement, unknown>();
    mockZoom.transform = vi.fn() as unknown as typeof mockZoom.transform;
    (viewer.zoomRef as unknown as { current: d3.ZoomBehavior<SVGSVGElement, unknown> }).current = mockZoom;

    const mockSvg = {
      getAttribute: vi.fn(() => "0"), // getAttribute returns 0
      node: () => ({}),
    } as unknown as SVGSVGElement;
    (viewer.svgRef as unknown as { current: SVGSVGElement }).current = mockSvg;

    // Instance explicitly has rendered dimensions
    viewer.renderedWidth = 800;
    viewer.renderedHeight = 400;

    const transform = viewer.zoomToImageRegion({ x: 450, y: 250, size: 100 });
    expect(transform).not.toBeNull();
    expect(transform?.k).toBe(7.5);
  });
});

describe("SVGViewer landmark label zoom counter-scaling", () => {
  it("tracks currentZoomK and safely handles zoom scale factors", () => {
    const props = createDefaultProps();
    const viewer = new SVGViewer(props);

    expect(viewer.currentZoomK).toBe(1.0);

    // When updating with k = 10 (e.g. toepad zoom), safeK is 10
    viewer.updateZoomDependentElements(10);
    expect(viewer.currentZoomK).toBe(10);

    // Clamps minimum safeK to 0.1
    viewer.updateZoomDependentElements(0.001);
    expect(viewer.currentZoomK).toBe(0.1);

    // Restores on normal zoom
    viewer.updateZoomDependentElements(2.5);
    expect(viewer.currentZoomK).toBe(2.5);
  });

  it("calculates correct counter-scaled SVG font-size and stroke-width for constant screen size", () => {
    const props = createDefaultProps();
    const viewer = new SVGViewer(props);

    const baseScreenFontSize = viewer.state.labelSize; // 8.5px
    const zoomScales = [1, 2, 5, 8.5, 10, 15];

    zoomScales.forEach((k) => {
      const svgFontSize = baseScreenFontSize / k;
      const screenRenderedFontSize = svgFontSize * k;
      expect(screenRenderedFontSize).toBeCloseTo(baseScreenFontSize);

      const svgStrokeWidth = 1.5 / k;
      const screenRenderedStrokeWidth = svgStrokeWidth * k;
      expect(screenRenderedStrokeWidth).toBeCloseTo(1.5);

      const textOffset = viewer.state.landmarkSize + 3 / k;
      // In SVG space, offset shrinks as k grows, so screen offset (offset * k) stays constant outside the dot
      const screenTextOffset = textOffset * k;
      expect(screenTextOffset).toBeCloseTo(viewer.state.landmarkSize * k + 3);
    });
  });
});
