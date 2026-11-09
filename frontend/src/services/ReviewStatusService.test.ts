import { describe, expect, it } from "vitest";
import type { BoundingBox } from "../models/AnnotationsData";
import { determineInitialReviewStatus } from "./ReviewStatusService";

describe("ReviewStatusService", () => {
  it("returns unreviewed for dorsal view with normal confidence boxes", () => {
    const boxes: BoundingBox[] = [
      { top: 10, left: 10, width: 50, height: 50, confidence: 0.95, label: "dorsal" },
    ];
    expect(determineInitialReviewStatus("dorsal", boxes)).toBe("unreviewed");
  });

  it("returns needs_review when any box has confidence below 0.50", () => {
    const boxes: BoundingBox[] = [
      { top: 10, left: 10, width: 50, height: 50, confidence: 0.92, label: "up_finger" },
      { top: 20, left: 20, width: 50, height: 50, confidence: 0.42, label: "up_toe" },
      { top: 30, left: 30, width: 50, height: 50, confidence: 0.88, label: "bot_finger" },
      { top: 40, left: 40, width: 50, height: 50, confidence: 0.85, label: "bot_toe" },
    ];
    expect(determineInitialReviewStatus("toepads", boxes)).toBe("needs_review");
  });

  it("returns needs_review in toepads view when fewer than 4 toepad boxes exist", () => {
    const boxes: BoundingBox[] = [
      { top: 10, left: 10, width: 50, height: 50, confidence: 0.9, label: "up_finger" },
      { top: 20, left: 20, width: 50, height: 50, confidence: 0.9, label: "up_toe" },
      { top: 30, left: 30, width: 50, height: 50, confidence: 0.9, label: "bot_finger" },
    ];
    expect(determineInitialReviewStatus("toepads", boxes)).toBe("needs_review");
  });

  it("returns unreviewed in toepads view when all 4 boxes have good confidence", () => {
    const boxes: BoundingBox[] = [
      { top: 10, left: 10, width: 50, height: 50, confidence: 0.85, label: "up_finger" },
      { top: 20, left: 20, width: 50, height: 50, confidence: 0.92, label: "up_toe" },
      { top: 30, left: 30, width: 50, height: 50, confidence: 0.88, label: "bot_finger" },
      { top: 40, left: 40, width: 50, height: 50, confidence: 0.91, label: "bot_toe" },
    ];
    expect(determineInitialReviewStatus("toepads", boxes)).toBe("unreviewed");
  });

  it("returns needs_review in toepads view if bounding boxes array is empty or undefined", () => {
    expect(determineInitialReviewStatus("toepads", [])).toBe("needs_review");
    expect(determineInitialReviewStatus("toepads", undefined)).toBe("needs_review");
  });

  it("returns unreviewed for non-toepad views without bounding boxes", () => {
    expect(determineInitialReviewStatus("dorsal", [])).toBe("unreviewed");
    expect(determineInitialReviewStatus("custom", undefined)).toBe("unreviewed");
  });
});
