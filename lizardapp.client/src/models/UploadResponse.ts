import type { Point } from "./Point";

export interface UploadResponse {
  name: string;
  coords: Point[];
}