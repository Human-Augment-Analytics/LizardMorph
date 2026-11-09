import type { BoundingBox } from "../models/AnnotationsData";
import type { ReviewStatus } from "../models/UploadHistoryItem";

/**
 * Determines the initial review status for a newly processed image.
 *
 * Rules:
 * - Low confidence (< 0.50) on any detected box -> 'needs_review'
 * - In 'toepads' view, if fewer than 4 digit boxes (up_finger, up_toe, bot_finger, bot_toe) are detected -> 'needs_review'
 * - Missing boxes entirely in 'toepads' view -> 'needs_review'
 * - Otherwise -> 'unreviewed'
 */
export function determineInitialReviewStatus(
  viewType: string | undefined,
  boundingBoxes?: BoundingBox[]
): ReviewStatus {
  if (!boundingBoxes || boundingBoxes.length === 0) {
    return viewType === "toepads" ? "needs_review" : "unreviewed";
  }

  // Any detection below 0.50 confidence flags the image for biologist review
  const hasLowConfidence = boundingBoxes.some(
    (b) => typeof b.confidence === "number" && b.confidence < 0.5
  );
  if (hasLowConfidence) {
    return "needs_review";
  }

  // In toepad mode, verify that all 4 digits were located
  if (viewType === "toepads") {
    const toepadBoxCount = boundingBoxes.filter((b) =>
      ["up_finger", "up_toe", "bot_finger", "bot_toe"].includes(
        (b.label || "").toLowerCase()
      )
    ).length;
    if (toepadBoxCount < 4) {
      return "needs_review";
    }
  }

  return "unreviewed";
}
