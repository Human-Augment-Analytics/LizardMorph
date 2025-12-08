import type { Point } from "./Point";

export interface BoundingBox {
  top: number;
  left: number;
  width: number;
  height: number;
  landmark_start_index?: number;
  landmark_count?: number;
}

export interface AnnotationsData {
  name: string;
  coords: Point[];
  bounding_boxes?: BoundingBox[];
  view_type?: string;
  toepad_predictor_type?: string;
}
