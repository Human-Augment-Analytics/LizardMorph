export interface Measurement {
  id: string;
  label: string;
  pointAId: number | null;
  pointBId: number | null;
  calculatedDistance: number | null; // Distance in the scale units
}

