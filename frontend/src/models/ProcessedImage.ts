import type { Point } from "../models/Point";
import type { ImageSet } from "../models/ImageSet";
import type { BoundingBox } from "./AnnotationsData";

export interface ProcessedImage {
  name: string;
  coords: Point[];
  originalCoords: Point[];
  imageSets: ImageSet;
  timestamp: string;
  boundingBoxes?: BoundingBox[];
}