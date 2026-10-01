import { describe, expect, it } from "vitest";

import type { BoundingBox } from "../models/AnnotationsData";
import type { Point } from "../models/Point";
import { TOEPAD_SLOTS, cropForBox, getToepadRegions } from "./ToepadRegions";

const box = (label: string, left: number, top: number, width: number, height: number): BoundingBox => ({
  label,
  left,
  top,
  width,
  height,
});

describe("getToepadRegions", () => {
  it("returns the four toepads in a fixed order and skips ruler and id boxes", () => {
    const boxes = [
      box("ruler", 0, 0, 200, 20),
      box("bot_toe", 400, 400, 40, 80),
      box("id", 10, 500, 60, 30),
      box("up_finger", 100, 100, 30, 10),
      box("bot_finger", 120, 300, 30, 10),
      box("up_toe", 400, 100, 40, 80),
    ];
    const points: Point[] = [
      { id: 0, x: 5, y: 5, box_idx: 0 },
      { id: 401, x: 410, y: 420, box_idx: 1 },
      { id: 402, x: 420, y: 460, box_idx: 1 },
      { id: 601, x: 110, y: 105, box_idx: 3 },
    ];

    const regions = getToepadRegions(points, boxes);

    expect(regions.map((r) => r.slot.label)).toEqual(TOEPAD_SLOTS.map((s) => s.label));
    expect(regions[0].points.map((p) => p.id)).toEqual([601]);
    expect(regions[3].points.map((p) => p.id)).toEqual([401, 402]);
    expect(regions[1].points).toEqual([]);
    expect(regions.every((r) => r.crop !== null)).toBe(true);
  });

  it("marks missing toepads as not detected", () => {
    const regions = getToepadRegions([], [box("up_toe", 0, 0, 10, 10)]);

    expect(regions.filter((r) => r.crop).map((r) => r.slot.label)).toEqual(["up_toe"]);
  });

  it("matches labels case-insensitively", () => {
    const regions = getToepadRegions([], [box("Bot_Finger", 0, 0, 10, 10)]);

    expect(regions.find((r) => r.slot.label === "bot_finger")?.crop).not.toBeNull();
  });

  it("falls back to landmarks inside the box when points have no box_idx", () => {
    const points: Point[] = [
      { id: 1, x: 105, y: 102 },
      { id: 2, x: 500, y: 500 },
    ];
    const regions = getToepadRegions(points, [box("up_finger", 100, 100, 30, 10)]);

    expect(regions[0].points.map((p) => p.id)).toEqual([1]);
  });
});

describe("cropForBox", () => {
  it("returns a padded square centred on the box", () => {
    const crop = cropForBox(box("up_toe", 100, 200, 40, 80));

    expect(crop.size).toBe(120);
    expect(crop.x + crop.size / 2).toBe(120);
    expect(crop.y + crop.size / 2).toBe(240);
  });

  it("enforces a minimum size for tiny boxes", () => {
    expect(cropForBox(box("up_finger", 0, 0, 4, 2)).size).toBe(48);
  });
});
