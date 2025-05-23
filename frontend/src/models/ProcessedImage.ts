import type { Point } from "../models/Point";
import type { ImageSet } from "../models/ImageSet";

export interface ProcessedImage {
  name: string;
  coords: Point[];
  originalCoords: Point[];
  imageSets: ImageSet;
  timestamp: string;
}