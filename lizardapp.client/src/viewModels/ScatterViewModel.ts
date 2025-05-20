import * as d3 from 'd3';
import type { Point } from '../models/Point';


export class ScatterViewModel {
  private imageFilename: string = '';
  private scatterData: Point[] = [];
  private originalData: Point[] = [];

  private scaleX: d3.ScaleLinear<number, number> | null = null;
  private scaleY: d3.ScaleLinear<number, number> | null = null;

  setImageFilename(name: string): void {
    this.imageFilename = name;
  }

  getImageFilename(): string {
    return this.imageFilename;
  }

  setScatterData(data: Point[]): void {
    this.scatterData = data;
    this.originalData = data.map(p => ({ ...p }));
  }

  getScatterData(): Point[] {
    return this.scatterData;
  }

  scaleData(svgWidth: number, svgHeight: number): void {
    this.scaleX = d3.scaleLinear()
      .domain([
        d3.min(this.scatterData, (d: Point) => d.x)!,
        d3.max(this.scatterData, (d: Point) => d.x)!
      ])
      .range([0, svgWidth]);

    this.scaleY = d3.scaleLinear()
      .domain([
        d3.min(this.scatterData, (d: Point) => d.y)!,
        d3.max(this.scatterData, (d: Point) => d.y)!
      ])
      .range([0, svgHeight]);

    this.scatterData = this.originalData.map(p => ({
      x: this.scaleX!(p.x),
      y: this.scaleY!(p.y),
    }));
  }

  getUnscaledData(): Point[] {
    return this.originalData;
  }
}
