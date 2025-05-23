import React, { Component, createRef } from "react";
import * as d3 from "d3";
import type { Point } from "../models/Point";
import { ApiService } from "../services/ApiService";
import { ScatterViewModel } from "../viewModels/ScatterViewModel";
import { ScatterViewModelStyles } from "./ScatterView.style";

interface ScatterViewState {
  scatterData: Point[];
  imageUrl: string | null;
  loading: boolean;
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
interface ScatterViewProps {}

export class ScatterView extends Component<ScatterViewProps, ScatterViewState> {
  readonly svgRef = createRef<SVGSVGElement>();
  readonly viewModel = new ScatterViewModel();

  state: ScatterViewState = {
    scatterData: [],
    imageUrl: null,
    loading: false,
  };

  handleUpload = async (
    e: React.ChangeEvent<HTMLInputElement>
  ): Promise<void> => {
    const file = e.target.files?.[0];
    if (!file) return;

    this.setState({ loading: true });

    const response = await ApiService.uploadImage(file);
    this.viewModel.setImageFilename(response.name);
    this.viewModel.setScatterData(response.coords);

    const blobUrl = await ApiService.fetchImageBlob(response.name);
    this.setState({ imageUrl: blobUrl }, this.renderSVG);
  };

  renderSVG = () => {
    const { imageUrl } = this.state;
    const svgElement = this.svgRef.current;

    if (!imageUrl || !svgElement) return;

    const img = new Image();
    img.src = imageUrl;

    img.onload = () => {
      const width = img.width;
      const height = img.height;

      this.viewModel.scaleData(width, height);
      this.setState({ scatterData: this.viewModel.getScatterData() });

      const svg = d3.select<SVGSVGElement, unknown>(svgElement);
      svg.selectAll("*").remove();
      svg.attr("width", width).attr("height", height);

      svg
        .append("image")
        .attr("href", imageUrl)
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", width)
        .attr("height", height);

      svg
        .append("g")
        .selectAll<SVGCircleElement, { x: number; y: number }>("circle")
        .data(this.state.scatterData)
        .enter()
        .append("circle")
        .attr("cx", (d) => d.x)
        .attr("cy", (d) => d.y)
        .attr("r", 3)
        .attr("fill", "red");

      const zoom = d3.zoom<SVGSVGElement, unknown>().on("zoom", (event) => {
        svg
          .selectAll<SVGGraphicsElement, unknown>("image, g")
          .attr("transform", event.transform.toString());
      });

      svg.call(zoom);
    };
  };

  handleDownload = async (): Promise<void> => {
    const unscaled = this.viewModel.getUnscaledData();
    const payload = {
      name: this.viewModel.getImageFilename(),
      coords: unscaled,
    };

    const response = await ApiService.postScatterData(payload);
    console.log("Saved:", response);
  };

  render() {
    return (
      <div style={ScatterViewModelStyles.scatterView}>
        <h2 style={ScatterViewModelStyles.heading}>Image Scatter App</h2>
        <input
          type="file"
          accept="image/*"
          onChange={this.handleUpload}
          style={ScatterViewModelStyles.fileInput}
        />
        <button
          onClick={this.handleDownload}
          style={ScatterViewModelStyles.button}
        >
          Download Points
        </button>
        <svg ref={this.svgRef} style={ScatterViewModelStyles.svgCanvas} />
      </div>
    );
  }
}
