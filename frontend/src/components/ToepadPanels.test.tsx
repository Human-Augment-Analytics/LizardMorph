import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { getToepadRegions } from "../services/ToepadRegions";

import type { BoundingBox } from "../models/AnnotationsData";
import type { Point } from "../models/Point";
import { ToepadPanels } from "./ToepadPanels";

const boxes: BoundingBox[] = [
  { label: "ruler", left: 0, top: 0, width: 200, height: 20 },
  { label: "up_finger", left: 100, top: 100, width: 30, height: 10 },
  { label: "bot_toe", left: 400, top: 400, width: 40, height: 80 },
];

const points: Point[] = [
  { id: 200, x: 105, y: 102, box_idx: 1 },
  { id: 201, x: 125, y: 108, box_idx: 1 },
  { id: 400, x: 410, y: 420, box_idx: 2 },
];

const render = (overrides: Partial<React.ComponentProps<typeof ToepadPanels>> = {}) =>
  renderToStaticMarkup(
    <ToepadPanels
      imageURL="/image.jpg"
      imageWidth={1440}
      imageHeight={900}
      points={points}
      boundingBoxes={boxes}
      selectedPoint={null}
      isEditMode={false}
      onToggleEditMode={() => {}}
      onPointSelect={() => {}}
      onPointsChange={() => {}}
      theme="light"
      {...overrides}
    />
  );

describe("ToepadPanels", () => {
  it("renders a card per toepad and marks undetected ones", () => {
    const markup = render();

    expect(markup).toContain("Upper Finger");
    expect(markup).toContain("Upper Toe");
    expect(markup).toContain("Lower Finger");
    expect(markup).toContain("Lower Toe");
    expect(markup).toContain("2 of 4 detected");
    expect(markup.match(/Not detected/g)).toHaveLength(2);
    expect(markup.match(/<circle /g)).toHaveLength(3);
  });

  it("zooms each close-up to its toepad box", () => {
    const markup = render();

    // up_finger box is 30x10 at (100,100): 48px square crop centred on (115,105)
    expect(markup).toContain('viewBox="91 81 48 48"');
  });

  it("highlights the selected landmark", () => {
    const markup = render({ selectedPoint: points[2] });

    expect(markup).toContain('fill="yellow"');
  });

  it("displays landmark numbers matching the big picture IDs (p.id + 1)", () => {
    const markup = render();

    // points have IDs 200, 201, 400 -> displayed as 201, 202, 401
    expect(markup).toContain(">201<");
    expect(markup).toContain(">202<");
    expect(markup).toContain(">401<");
  });

  it("tells the user how to edit, and how to drag once editing", () => {
    expect(render()).toContain("Click Edit Points to adjust landmarks");
    expect(render({ isEditMode: true })).toContain("Drag a landmark to move it");
  });

  it("still shows the grid when no toepads were detected", () => {
    const markup = render({ boundingBoxes: [boxes[0]] });

    expect(markup).toContain("0 of 4 detected");
    expect(markup.match(/Not detected/g)).toHaveLength(4);
  });

  it("renders nothing until the image has loaded", () => {
    expect(render({ imageWidth: 0, imageHeight: 0 })).toBe("");
  });

  it("defaults the controls and shortcuts key to closed view", () => {
    const markup = render();
    expect(markup).toContain("Controls &amp; Shortcuts Key");
    expect(markup).toContain("Show ▼");
    expect(markup).not.toContain("Mouse &amp; Touch Controls");
  });

  it("renders the controls and shortcuts key tables when opened", () => {
    const markup = render({ defaultLegendOpen: true });
    expect(markup).toContain("Controls &amp; Shortcuts Key");
    expect(markup).toContain("Hide ▲");
    expect(markup).toContain("Mouse &amp; Touch Controls");
    expect(markup).toContain("Keyboard Shortcuts");
    expect(markup).toContain("Ctrl+Z");
    expect(markup).toContain("Right-Click");
    expect(markup).toContain("Toggle bounding boxes");
  });

  it("renders a '⤢ Zoom to Toe' button on detected toepad cards", () => {
    const markup = render();
    expect(markup).toContain("Zoom to Toe");
    expect(markup).toContain('title="Zoom in on Upper Finger"');
    expect(markup).toContain('title="Zoom in on Lower Toe"');
  });

  it("calls onZoomToToe with region when handleToepadClick is called", () => {
    const onZoomToToe = vi.fn();
    const panels = new ToepadPanels({
      imageURL: "/image.jpg",
      imageWidth: 1440,
      imageHeight: 900,
      points,
      boundingBoxes: boxes,
      selectedPoint: null,
      isEditMode: false,
      onToggleEditMode: () => {},
      onPointSelect: () => {},
      onPointsChange: () => {},
      onZoomToToe,
      theme: "light",
    });

    const regions = getToepadRegions(points, boxes);
    const upFinger = regions.find((r) => r.slot.label === "up_finger")!;
    panels.handleToepadClick(upFinger);

    expect(onZoomToToe).toHaveBeenCalledWith(upFinger);
    expect(panels.state.activeModalSlotLabel).toBeNull();
  });

  it("highlights active toepad card and changes button to '↺ Reset Zoom' when activeSlotLabel matches", () => {
    const markup = render({ activeSlotLabel: "up_finger" });
    expect(markup).toContain("Reset Zoom (Upper Finger)");
    expect(markup).toContain("↺ Reset Zoom");
    expect(markup).toContain("border-color:#00E5FF");
  });

  it("does not render ToepadModal popup when onZoomToToe is provided even if activeModalSlotLabel is set", () => {
    const markup = render({
      defaultActiveModalSlotLabel: "up_finger",
      onZoomToToe: () => {},
    });
    expect(markup).not.toContain('role="dialog"');
    expect(markup).not.toContain("Upper Finger Close-up");
  });

  it("renders ToepadModal popup when activeModalSlotLabel is set", () => {
    const markup = render({ defaultActiveModalSlotLabel: "up_finger" });
    expect(markup).toContain('role="dialog"');
    expect(markup).toContain('aria-modal="true"');
    expect(markup).toContain("Upper Finger Close-up");
    expect(markup).toContain("✕ Close");
  });

  it("opens modal via openModal and closes via closeModal", () => {
    const panels = new ToepadPanels({
      imageURL: "/image.jpg",
      imageWidth: 1440,
      imageHeight: 900,
      points,
      boundingBoxes: boxes,
      selectedPoint: null,
      isEditMode: false,
      onToggleEditMode: () => {},
      onPointSelect: () => {},
      onPointsChange: () => {},
      theme: "light",
    });

    expect(panels.state.activeModalSlotLabel).toBeNull();
    panels.setState = (updater) => {
      const next = typeof updater === "function" ? updater(panels.state, panels.props) : updater;
      Object.assign(panels.state, next);
    };

    // Open modal
    panels.openModal("up_finger");
    expect(panels.state.activeModalSlotLabel).toBe("up_finger");

    const openMarkup = renderToStaticMarkup(panels.render());
    expect(openMarkup).toContain('role="dialog"');
    expect(openMarkup).toContain("Upper Finger Close-up");

    // Close modal
    panels.closeModal();
    expect(panels.state.activeModalSlotLabel).toBeNull();

    const closedMarkup = renderToStaticMarkup(panels.render());
    expect(closedMarkup).not.toContain('role="dialog"');
  });
});
