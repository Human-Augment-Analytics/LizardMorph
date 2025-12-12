export interface ScaleSettings {
  pointAId: number | null;
  pointBId: number | null;
  value: number | null;
  units: string;
}

export const UNITS = [
  { value: "mm", label: "mm" },
  { value: "cm", label: "cm" },
  { value: "m", label: "m" },
  { value: "inch", label: "inch" },
  { value: "ft", label: "ft" },
  { value: "pixels", label: "pixels" },
] as const;

