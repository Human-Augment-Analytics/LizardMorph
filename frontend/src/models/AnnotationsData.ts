import type { Point } from "./Point";

export interface BoundingBox {
  top: number;
  left: number;
  width: number;
  height: number;
  landmark_start_index?: number;
  landmark_count?: number;
  id?: number | string;
  label?: string;
  confidence?: number;
  obb_corners?: { x: number, y: number }[];
}

export interface AnnotationsData {
  name: string;
  coords: Point[];
  bounding_boxes?: BoundingBox[];
  view_type?: string;
  toepad_predictor_type?: string;
  error?: string;
  session_id?: string;
}
