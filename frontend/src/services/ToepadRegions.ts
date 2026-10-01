import type { Point } from "../models/Point";
import type { BoundingBox } from "../models/AnnotationsData";

export interface ToepadSlot {
  label: "up_finger" | "up_toe" | "bot_finger" | "bot_toe";
  title: string;
  /** Matches the bounding-box colour used for this class in SVGViewer. */
  color: string;
}

export const TOEPAD_SLOTS: readonly ToepadSlot[] = [
  { label: "up_finger", title: "Upper Finger", color: "#00cc00" },
  { label: "up_toe", title: "Upper Toe", color: "#0066dd" },
  { label: "bot_finger", title: "Lower Finger", color: "#00ff00" },
  { label: "bot_toe", title: "Lower Toe", color: "#0088ff" },
];

export interface CropRect {
  x: number;
  y: number;
  size: number;
}

export interface ToepadRegion {
  slot: ToepadSlot;
  /** Null when the detector found no box of this class. */
  crop: CropRect | null;
  /** Landmarks in image-pixel space, in predictor order. */
  points: Point[];
}

const CROP_PADDING = 1.5;
const MIN_CROP_SIZE = 48;

/** Square crop centred on the box, padded so the whole digit and some context are visible. */
export function cropForBox(box: BoundingBox): CropRect {
  const size = Math.max(Math.max(box.width, box.height) * CROP_PADDING, MIN_CROP_SIZE);
  const cx = box.left + box.width / 2;
  const cy = box.top + box.height / 2;
  return { x: cx - size / 2, y: cy - size / 2, size };
}

function pointsForBox(points: Point[], boxIdx: number, box: BoundingBox): Point[] {
  const tagged = points.filter((p) => p.box_idx === boxIdx);
  if (tagged.length > 0) return tagged;
  // Older payloads carry no box_idx; fall back to the landmarks inside the box.
  return points.filter(
    (p) =>
      p.box_idx === undefined &&
      p.x >= box.left &&
      p.x <= box.left + box.width &&
      p.y >= box.top &&
      p.y <= box.top + box.height
  );
}

/** One region per toepad slot, in TOEPAD_SLOTS order. */
export function getToepadRegions(points: Point[], boxes: BoundingBox[]): ToepadRegion[] {
  return TOEPAD_SLOTS.map((slot) => {
    const boxIdx = boxes.findIndex((b) => (b.label || "").toLowerCase() === slot.label);
    if (boxIdx < 0) return { slot, crop: null, points: [] };
    const box = boxes[boxIdx];
    return {
      slot,
      crop: cropForBox(box),
      points: pointsForBox(points, boxIdx, box),
    };
  });
}
