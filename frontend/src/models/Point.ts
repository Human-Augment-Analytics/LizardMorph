export interface Point {
  x: number;
  y: number;
  id: number;
  /** Index into the image's bounding boxes, set for multi-object predictions (e.g. toepads). */
  box_idx?: number;
  landmark_id?: number;
}