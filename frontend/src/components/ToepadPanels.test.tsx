import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

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
});
