import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { ToepadRegion } from "../services/ToepadRegions";
import type { Point } from "../models/Point";
import { ToepadModal } from "./ToepadModal";

const sampleRegion: ToepadRegion = {
  slot: { label: "up_finger", title: "Upper Finger", color: "#00cc00" },
  crop: { x: 100, y: 100, size: 50 },
  points: [
    { id: 200, x: 110, y: 115, box_idx: 1 },
    { id: 201, x: 130, y: 125, box_idx: 1 },
  ],
};

const allSpecimenPoints: Point[] = [
  { id: 100, x: 50, y: 50 },
  { id: 200, x: 110, y: 115, box_idx: 1 },
  { id: 201, x: 130, y: 125, box_idx: 1 },
  { id: 300, x: 500, y: 500 },
];

const createDefaultProps = (
  overrides: Partial<React.ComponentProps<typeof ToepadModal>> = {}
) => ({
  region: sampleRegion,
  allPoints: allSpecimenPoints,
  imageURL: "/test_image.jpg",
  imageWidth: 1000,
  imageHeight: 800,
  selectedPoint: null,
  isEditMode: false,
  onToggleEditMode: vi.fn(),
  onPointSelect: vi.fn(),
  onPointsChange: vi.fn(),
  onClose: vi.fn(),
  theme: "light" as const,
  ...overrides,
});

describe("ToepadModal", () => {
  it("renders enlarged toepad title, close button, edit mode toggle, and zoom controls", () => {
    const props = createDefaultProps();
    const markup = renderToStaticMarkup(<ToepadModal {...props} />);

    // Header & Title
    expect(markup).toContain("Upper Finger Close-up");
    expect(markup).toContain('id="toepad-modal-title"');
    expect(markup).toContain("2 pts");

    // Edit points toggle button
    expect(markup).toContain("✏️ Edit Points: Off");

    // Zoom controls
    expect(markup).toContain("+ Zoom In");
    expect(markup).toContain("- Zoom Out");
    expect(markup).toContain("↺ Reset");
    expect(markup).toContain("100%");

    // Close button
    expect(markup).toContain("✕ Close");

    // Accessibility attributes
    expect(markup).toContain('role="dialog"');
    expect(markup).toContain('aria-modal="true"');
    expect(markup).toContain('aria-labelledby="toepad-modal-title"');
  });

  it("reflects active Edit Mode in toggle button and overlay cursor", () => {
    const props = createDefaultProps({ isEditMode: true });
    const markup = renderToStaticMarkup(<ToepadModal {...props} />);

    expect(markup).toContain("✏️ Edit Points: On");
    expect(markup).toContain("cursor:crosshair");
  });

  it("renders landmark labels matching specimen IDs (p.id + 1) and highlights selected landmark", () => {
    const props = createDefaultProps({
      selectedPoint: sampleRegion.points[0],
    });
    const markup = renderToStaticMarkup(<ToepadModal {...props} />);

    // IDs 200 and 201 rendered as 201 and 202
    expect(markup).toContain(">201<");
    expect(markup).toContain(">202<");

    // Selected point is yellow, unselected is red
    expect(markup).toContain('fill="yellow"');
    expect(markup).toContain('fill="red"');
  });

  it("renders empty state when crop or imageURL is absent", () => {
    const unlocatedRegion: ToepadRegion = {
      slot: { label: "bot_toe", title: "Lower Toe", color: "#0088ff" },
      crop: null,
      points: [],
    };
    const markup = renderToStaticMarkup(
      <ToepadModal {...createDefaultProps({ region: unlocatedRegion })} />
    );

    expect(markup).toContain("Not detected");
  });

  it("closes modal when Escape key is pressed", () => {
    const onClose = vi.fn();
    const modal = new ToepadModal(createDefaultProps({ onClose }));

    const preventDefault = vi.fn();
    modal.handleKeyDown({
      key: "Escape",
      preventDefault,
    } as unknown as KeyboardEvent);

    expect(preventDefault).toHaveBeenCalled();
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not close modal on unrelated key presses", () => {
    const onClose = vi.fn();
    const modal = new ToepadModal(createDefaultProps({ onClose }));

    modal.handleKeyDown({
      key: "Enter",
      preventDefault: vi.fn(),
    } as unknown as KeyboardEvent);

    expect(onClose).not.toHaveBeenCalled();
  });

  it("closes modal when clicking the backdrop, but not when clicking inside dialog", () => {
    const onClose = vi.fn();
    const modal = new ToepadModal(createDefaultProps({ onClose }));

    const backdropEl = { id: "backdrop" };
    const dialogEl = { id: "dialog" };

    // Click directly on backdrop
    modal.handleBackdropClick({
      target: backdropEl,
      currentTarget: backdropEl,
    } as unknown as React.MouseEvent<HTMLDivElement>);
    expect(onClose).toHaveBeenCalledTimes(1);

    // Click on inner dialog child
    modal.handleBackdropClick({
      target: dialogEl,
      currentTarget: backdropEl,
    } as unknown as React.MouseEvent<HTMLDivElement>);
    expect(onClose).toHaveBeenCalledTimes(1); // Still 1
  });

  it("handles zoom in, zoom out, and reset zoom", () => {
    const modal = new ToepadModal(createDefaultProps());
    expect(modal.state.zoom).toBe(1);

    modal.setState = vi.fn().mockImplementation((updater) => {
      const next = typeof updater === "function" ? updater(modal.state, modal.props) : updater;
      Object.assign(modal.state, next);
    });

    modal.handleZoomIn();
    expect(modal.state.zoom).toBeGreaterThan(1);

    modal.handleZoomOut();
    expect(modal.state.zoom).toBe(1);

    modal.handleZoomIn();
    modal.handleZoomIn();
    expect(modal.state.zoom).toBeGreaterThan(1.4);

    modal.handleResetZoom();
    expect(modal.state.zoom).toBe(1);
    expect(modal.state.pan).toEqual({ x: 0, y: 0 });
  });

  it("handles wheel zoom on viewport", () => {
    const modal = new ToepadModal(createDefaultProps());
    modal.setState = vi.fn().mockImplementation((updater) => {
      const next = typeof updater === "function" ? updater(modal.state, modal.props) : updater;
      Object.assign(modal.state, next);
    });

    const preventDefault = vi.fn();
    const stopPropagation = vi.fn();

    // Wheel down -> zoom out
    modal.handleWheel({
      deltaY: 100,
      preventDefault,
      stopPropagation,
    } as unknown as React.WheelEvent<HTMLDivElement>);

    expect(preventDefault).toHaveBeenCalled();
    expect(modal.state.zoom).toBeLessThan(1);

    // Wheel up -> zoom in
    modal.handleWheel({
      deltaY: -100,
      preventDefault,
      stopPropagation,
    } as unknown as React.WheelEvent<HTMLDivElement>);

    expect(modal.state.zoom).toBeGreaterThan(0.8);
  });

  it("pans canvas in view mode and with Ctrl in edit mode", () => {
    // In view mode
    const viewModal = new ToepadModal(createDefaultProps({ isEditMode: false }));
    viewModal.setState = vi.fn().mockImplementation((updater) => {
      const next = typeof updater === "function" ? updater(viewModal.state, viewModal.props) : updater;
      Object.assign(viewModal.state, next);
    });

    const setPointerCapture = vi.fn();
    viewModal.handleSvgPointerDown({
      button: 0,
      clientX: 50,
      clientY: 50,
      preventDefault: vi.fn(),
      currentTarget: { setPointerCapture } as unknown as SVGSVGElement,
    } as unknown as React.PointerEvent<SVGSVGElement>);

    expect(viewModal.state.isPanning).toBe(true);

    viewModal.handlePointerMove({
      clientX: 70,
      clientY: 85,
    } as unknown as React.PointerEvent<SVGSVGElement>);

    expect(viewModal.state.pan).toEqual({ x: 20, y: 35 });

    const releasePointerCapture = vi.fn();
    viewModal.handlePointerUp({
      currentTarget: {
        hasPointerCapture: () => true,
        releasePointerCapture,
      } as unknown as SVGSVGElement,
    } as unknown as React.PointerEvent<SVGSVGElement>);

    expect(viewModal.state.isPanning).toBe(false);

    // In edit mode without Ctrl: panning does NOT start
    const editModal = new ToepadModal(createDefaultProps({ isEditMode: true }));
    editModal.setState = vi.fn().mockImplementation((updater) => {
      const next = typeof updater === "function" ? updater(editModal.state, editModal.props) : updater;
      Object.assign(editModal.state, next);
    });

    editModal.handleSvgPointerDown({
      button: 0,
      clientX: 50,
      clientY: 50,
      ctrlKey: false,
      metaKey: false,
      preventDefault: vi.fn(),
    } as unknown as React.PointerEvent<SVGSVGElement>);

    expect(editModal.state.isPanning).toBe(false);

    // In edit mode with Ctrl: panning DOES start
    editModal.handleSvgPointerDown({
      button: 0,
      clientX: 50,
      clientY: 50,
      ctrlKey: true,
      metaKey: false,
      preventDefault: vi.fn(),
      currentTarget: { setPointerCapture } as unknown as SVGSVGElement,
    } as unknown as React.PointerEvent<SVGSVGElement>);

    expect(editModal.state.isPanning).toBe(true);
  });

  it("drags landmark inside modal in Edit Mode and emits onPointsChange on release", () => {
    const onPointsChange = vi.fn();
    const onPointSelect = vi.fn();
    const modal = new ToepadModal(
      createDefaultProps({
        isEditMode: true,
        onPointsChange,
        onPointSelect,
      })
    );

    modal.setState = vi.fn().mockImplementation((updater) => {
      const next = typeof updater === "function" ? updater(modal.state, modal.props) : updater;
      Object.assign(modal.state, next);
    });

    const targetPoint = sampleRegion.points[0]; // id: 200, x: 110, y: 115
    const setPointerCapture = vi.fn();

    // 1. Pointer down on landmark
    modal.handlePointPointerDown(
      {
        button: 0,
        clientX: 100,
        clientY: 100,
        preventDefault: vi.fn(),
        stopPropagation: vi.fn(),
        currentTarget: {
          ownerSVGElement: { setPointerCapture },
        } as unknown as SVGGElement,
      } as unknown as React.PointerEvent<SVGGElement>,
      targetPoint
    );

    expect(onPointSelect).toHaveBeenCalledWith(targetPoint);
    expect(modal.state.drag).toBeDefined();
    expect(modal.state.drag?.id).toBe(200);
    expect(modal.state.undoStack).toHaveLength(1);

    // Mock SVG CTM mapping or bounding rect for toImagePoint
    const mockSvg = {
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 500, height: 500 }),
      hasPointerCapture: () => true,
      releasePointerCapture: vi.fn(),
    } as unknown as SVGSVGElement;

    // 2. Pointer move beyond threshold (normX = 200 / 500 = 0.4 -> x = 100 + 0.4*50 = 120)
    modal.handlePointerMove({
      clientX: 200,
      clientY: 250,
      currentTarget: mockSvg,
    } as unknown as React.PointerEvent<SVGSVGElement>);

    expect(modal.state.drag?.moved).toBe(true);
    expect(modal.state.drag?.x).toBe(120);
    expect(modal.state.drag?.y).toBe(125);

    // 3. Pointer up -> commits changes
    modal.handlePointerUp({
      currentTarget: mockSvg,
    } as unknown as React.PointerEvent<SVGSVGElement>);

    expect(onPointsChange).toHaveBeenCalledTimes(1);
    const updated = onPointsChange.mock.calls[0][0] as Point[];
    const movedPoint = updated.find((p) => p.id === 200);
    expect(movedPoint).toEqual({ id: 200, x: 120, y: 125, box_idx: 1 });

    // Other points unchanged
    expect(updated.find((p) => p.id === 100)?.x).toBe(50);
    expect(updated.find((p) => p.id === 300)?.x).toBe(500);
    expect(modal.state.drag).toBeNull();
  });

  it("supports Ctrl+Z undo inside modal to revert landmark moves", () => {
    const onPointsChange = vi.fn();
    const modal = new ToepadModal(
      createDefaultProps({
        isEditMode: true,
        onPointsChange,
      })
    );

    modal.setState = vi.fn().mockImplementation((updater) => {
      const next = typeof updater === "function" ? updater(modal.state, modal.props) : updater;
      Object.assign(modal.state, next);
    });

    // Populate undoStack with initial points
    Object.assign(modal.state, { undoStack: [[...allSpecimenPoints]] });

    const preventDefault = vi.fn();
    modal.handleKeyDown({
      key: "z",
      ctrlKey: true,
      metaKey: false,
      preventDefault,
    } as unknown as KeyboardEvent);

    expect(preventDefault).toHaveBeenCalled();
    expect(onPointsChange).toHaveBeenCalledWith(allSpecimenPoints);
    expect(modal.state.undoStack).toHaveLength(0);
  });

  it("toggles edit mode on right-click", () => {
    const onToggleEditMode = vi.fn();
    const modal = new ToepadModal(createDefaultProps({ onToggleEditMode }));

    const preventDefault = vi.fn();
    modal.handleContextMenu({
      preventDefault,
    } as unknown as React.MouseEvent);

    expect(preventDefault).toHaveBeenCalled();
    expect(onToggleEditMode).toHaveBeenCalledTimes(1);
  });

  it("adapts styling tokens for light and dark theme", () => {
    const lightMarkup = renderToStaticMarkup(
      <ToepadModal {...createDefaultProps({ theme: "light" })} />
    );
    expect(lightMarkup).toContain("background:#ffffff");

    const darkMarkup = renderToStaticMarkup(
      <ToepadModal {...createDefaultProps({ theme: "dark" })} />
    );
    expect(darkMarkup).toContain("background:#17202e");
  });

  it("renders Undo button in header controls with accessible attributes", () => {
    const markup = renderToStaticMarkup(<ToepadModal {...createDefaultProps()} />);
    expect(markup).toContain("↶ Undo");
    expect(markup).toContain('aria-label="Undo landmark move"');
    expect(markup).toContain("disabled=");
  });

  it("renders accessible dialog container with tabIndex -1 and aria attributes", () => {
    const markup = renderToStaticMarkup(<ToepadModal {...createDefaultProps()} />);
    expect(markup).toContain('tabindex="-1"');
    expect(markup).toContain('role="dialog"');
    expect(markup).toContain('aria-modal="true"');
    expect(markup).toContain('aria-labelledby="toepad-modal-title"');
  });

  it("renders landmark labels with stroke outline and paint-order for contrast", () => {
    const markup = renderToStaticMarkup(<ToepadModal {...createDefaultProps()} />);
    expect(markup).toContain('paint-order="stroke fill"');
    expect(markup).toContain('stroke="rgba(0,0,0,0.85)"');
  });

  it("disables zoom buttons at min and max limits", () => {
    const minZoomMarkup = renderToStaticMarkup(
      <ToepadModal {...createDefaultProps({ initialZoom: 0.5 })} />
    );
    expect(minZoomMarkup).toMatch(/<button[^>]*disabled=""[^>]*aria-label="Zoom Out"/);

    const maxZoomMarkup = renderToStaticMarkup(
      <ToepadModal {...createDefaultProps({ initialZoom: 10 })} />
    );
    expect(maxZoomMarkup).toMatch(/<button[^>]*disabled=""[^>]*aria-label="Zoom In"/);
  });
});

