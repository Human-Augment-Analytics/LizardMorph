import { Component, createRef } from "react";
import * as d3 from "d3";
import { SVGViewerStyles } from "./SVGViewer.style";
import type { Point } from "../models/Point";
import type { UploadHistoryItem } from "../models/UploadHistoryItem";

interface SVGViewerProps {
  dataFetched: boolean;
  loading: boolean;
  dataLoading: boolean;
  dataError: Error | null;
  uploadHistory: UploadHistoryItem[];
  scatterData: Point[];
  originalScatterData: Point[];
  selectedPoint: Point | null;
  needsScaling: boolean;
  currentImageURL: string | null;
  imageWidth: number;
  imageHeight: number;
  zoomTransform: d3.ZoomTransform;
  onPointSelect: (point: Point) => void;
  onScatterDataUpdate: (
    scatterData: Point[],
    originalScatterData: Point[]
  ) => void;
  onScalingComplete: () => void;
  onZoomChange: (transform: d3.ZoomTransform) => void;
  isEditMode: boolean;
  onToggleEditMode: () => void;
  onResetZoom: () => void;
}

export class SVGViewer extends Component<SVGViewerProps, object> {
  readonly svgRef = createRef<SVGSVGElement>();
  readonly zoomRef = createRef<d3.ZoomBehavior<SVGSVGElement, unknown>>();

  componentDidUpdate(prevProps: SVGViewerProps) {
    if (
      this.props.currentImageURL &&
      this.props.imageWidth &&
      this.props.imageHeight &&
      this.props.originalScatterData.length > 0 &&
      (prevProps.currentImageURL !== this.props.currentImageURL ||
        prevProps.imageWidth !== this.props.imageWidth ||
        prevProps.imageHeight !== this.props.imageHeight ||
        this.props.needsScaling)
    ) {
      setTimeout(() => {
        this.renderSVG();
      }, 0);
    }

    // Update drag behavior if edit mode changed
    if (prevProps.isEditMode !== this.props.isEditMode) {
      this.updateDragBehavior();
    }

    if (prevProps.selectedPoint !== this.props.selectedPoint) {
      this.updatePointSelection();
    }

    // Apply zoom transform if it changed and not in edit mode
    if (
      prevProps.zoomTransform !== this.props.zoomTransform &&
      !this.props.isEditMode &&
      this.svgRef.current &&
      this.zoomRef.current
    ) {
      const svg = d3.select(this.svgRef.current);
      svg.call(this.zoomRef.current.transform, this.props.zoomTransform);
    }
  }

  private readonly renderSVG = (): void => {
    if (
      this.props.currentImageURL &&
      this.props.imageWidth &&
      this.props.imageHeight &&
      this.props.originalScatterData.length > 0
    ) {
      console.log(
        "Rendering SVG with image dimensions:",
        this.props.imageWidth,
        "x",
        this.props.imageHeight
      );

      const svg = d3.select<SVGSVGElement, unknown>(this.svgRef.current!);
      svg.selectAll("*").remove(); // Clear SVG first to prevent duplication

      // Calculate dimensions maintaining aspect ratio
      const windowHeight = window.innerHeight - window.innerHeight * 0.2;
      const width =
        windowHeight * (this.props.imageWidth / this.props.imageHeight);
      const height = windowHeight;

      svg.attr("width", width).attr("height", height);

      // Calculate scaling factors
      const xScale = width / this.props.imageWidth;
      const yScale = height / this.props.imageHeight;

      console.log("Scaling factors:", xScale, yScale);

      // Define scaling functions
      if (this.props.needsScaling) {
        console.log("Scaling scatter data to match SVG dimensions");

        // Only calculate scale once per image load
        const scaleX = d3
          .scaleLinear()
          .domain([0, this.props.imageWidth])
          .range([0, width]);

        const scaleY = d3
          .scaleLinear()
          .domain([0, this.props.imageHeight])
          .range([0, height]); // Scale the data
        const scaledData = this.props.originalScatterData.map(
          (point: Point) => {
            if (typeof point.x === "number" && typeof point.y === "number") {
              return {
                ...point,
                x: scaleX(point.x),
                y: scaleY(point.y),
              };
            } else {
              console.error("Invalid point data:", point);
              return { ...point, x: 0, y: 0 };
            }
          }
        );

        if (scaledData.length > 0) {
          console.log(
            "First original point:",
            this.props.originalScatterData[0]
          );
          console.log("First scaled data point:", scaledData[0]);
        }

        // Update parent component with scaled data
        this.props.onScatterDataUpdate(
          scaledData,
          this.props.originalScatterData
        );
        this.props.onScalingComplete();
      }

      // Create container for zoom behavior
      const zoomContainer = svg.append("g").attr("class", "zoom-container");

      // Add image to the zoom container
      zoomContainer
        .append("image")
        .attr("class", "background-img")
        .attr("href", this.props.currentImageURL)
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", width)
        .attr("height", height)
        .attr("preserveAspectRatio", "xMidYMid slice");

      // Add scatter points to the zoom container
      const scatterPlotGroup = zoomContainer
        .append("g")
        .attr("class", "scatter-points");

      const pointGroups = scatterPlotGroup
        .selectAll("g")
        .data(this.props.scatterData)
        .enter()
        .append("g");

      pointGroups.each((d, i, nodes) => {
        const g = d3.select(nodes[i]);

        // Add the point
        g.append("circle")
          .attr("cx", d.x)
          .attr("cy", d.y)
          .attr("r", 3)
          .attr(
            "fill",
            this.props.selectedPoint && d.id === this.props.selectedPoint.id
              ? "yellow"
              : "red"
          )
          .attr("stroke", "black")
          .attr(
            "stroke-width",
            this.props.selectedPoint && d.id === this.props.selectedPoint.id
              ? 2
              : 1
          )
          .attr("data-id", d.id)
          .style("cursor", "pointer");

        // Add the number label
        g.append("text")
          .attr("x", d.x + 5)
          .attr("y", d.y - 5)
          .text(d.id)
          .attr("font-size", "10px")
          .attr("fill", "white")
          .attr("stroke", "black")
          .attr("stroke-width", "0.5px");
      });

      // Add drag behavior only if in edit mode
      if (this.props.isEditMode) {
        pointGroups.call(
          d3
            .drag<SVGGElement, Point>()
            .on("start", this.dragstarted)
            .on("drag", this.dragged)
            .on("end", this.dragended)
        );
      }

      // Add zoom behavior, preserving the current zoom state
      const zoom = d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.5, 5])
        .on("zoom", (event) => {
          zoomContainer.attr("transform", event.transform.toString());
          this.props.onZoomChange(event.transform);
        })
        .filter((event: Event) => {
          const mouseEvent = event as MouseEvent;
          if (event.type === "dblclick") {
            event.preventDefault();
            return false;
          }
          // Disable zoom/pan when in edit mode
          if (this.props.isEditMode) {
            return false;
          }
          return !mouseEvent.button && event.type !== "dblclick";
        });

      // Store the zoom reference for external control
      this.zoomRef.current = zoom;

      // Apply zoom behavior to the SVG only if not in edit mode
      if (!this.props.isEditMode) {
        svg.call(zoom);
      }

      // Always apply the stored transform to preserve zoom state
      if (
        this.props.zoomTransform &&
        this.props.zoomTransform !== d3.zoomIdentity &&
        !this.props.isEditMode
      ) {
        svg.call(zoom.transform, this.props.zoomTransform);
      }
    }
  };

  private readonly updatePointSelection = (): void => {
    if (this.svgRef.current && this.props.scatterData.length > 0) {
      const svg = d3.select(this.svgRef.current);
      svg
        .selectAll<SVGCircleElement, Point>("circle")
        .attr("fill", (d: Point) => {
          const id = d.id ?? d3.select(svg.node()).attr("data-id");
          return this.props.selectedPoint && id == this.props.selectedPoint.id
            ? "yellow"
            : "red";
        })
        .attr("stroke-width", (d: Point) => {
          const id = d.id ?? d3.select(svg.node()).attr("data-id");
          return this.props.selectedPoint && id == this.props.selectedPoint.id
            ? 2
            : 1;
        });
    }
  };

  private readonly dragstarted = (
    event: d3.D3DragEvent<SVGGElement, Point, Point>,
    d: Point
  ): void => {
    event.sourceEvent.stopPropagation();
    d3.select(event.sourceEvent.target.parentNode)
      .raise()
      .attr("stroke", "black");
    this.props.onPointSelect(d);
    const svg = d3.select(this.svgRef.current);
    const scatterPlotGroup = svg.select<SVGGElement>(".scatter-points");
    scatterPlotGroup
      .selectAll<SVGCircleElement, Point>("circle")
      .attr("fill", (p: Point) => (p.id === d.id ? "yellow" : "red"))
      .attr("stroke-width", (p: Point) => (p.id === d.id ? 2 : 1));
  };

  private readonly dragged = (
    event: d3.D3DragEvent<SVGGElement, Point, Point>,
    d: Point
  ): void => {
    const point = d3.pointer(event, this.svgRef.current);
    const transform = d3.zoomTransform(this.svgRef.current!);

    const x = (point[0] - transform.x) / transform.k;
    const y = (point[1] - transform.y) / transform.k;

    const group = d3.select(event.sourceEvent.target.parentNode);
    group.select("circle").attr("cx", x).attr("cy", y);

    group
      .select("text")
      .attr("x", x + 5)
      .attr("y", y - 5);

    const updatedScatterData = this.props.scatterData.map((p) =>
      p.id === d.id ? { ...p, x, y } : p
    );

    const scaleX = d3
      .scaleLinear()
      .domain([0, this.props.imageWidth])
      .range([
        0,
        window.innerHeight * (this.props.imageWidth / this.props.imageHeight),
      ]);

    const scaleY = d3
      .scaleLinear()
      .domain([0, this.props.imageHeight])
      .range([0, window.innerHeight - window.innerHeight * 0.2]);

    const updatedOriginalCoords = this.props.originalScatterData.map(
      (p: Point) =>
        p.id === d.id
          ? {
              ...p,
              x: scaleX.invert(x),
              y: scaleY.invert(y),
            }
          : p
    );

    this.props.onScatterDataUpdate(updatedScatterData, updatedOriginalCoords);
    this.props.onPointSelect({ ...d, x, y });
  };

  private readonly dragended = (
    event: d3.D3DragEvent<SVGGElement, Point, Point>
  ): void => {
    event.sourceEvent.stopPropagation();
    d3.select(event.sourceEvent.target.parentNode).attr("stroke", "black");
  };

  // Add or remove drag behavior on points depending on edit mode
  private readonly updateDragBehavior = (): void => {
    if (!this.svgRef.current) return;
    const svg = d3.select(this.svgRef.current);
    const scatterPlotGroup = svg.select<SVGGElement>(".scatter-points");
    if (scatterPlotGroup.empty()) return;
    const pointGroups = scatterPlotGroup.selectAll<SVGGElement, Point>("g");
    if (this.props.isEditMode) {
      pointGroups.call(
        d3
          .drag<SVGGElement, Point>()
          .on("start", this.dragstarted)
          .on("drag", this.dragged)
          .on("end", this.dragended)
      );
    } else {
      pointGroups.call(
        d3
          .drag<SVGGElement, Point>()
          .on("start", null)
          .on("drag", null)
          .on("end", null)
      );
    }
  };

  render() {
    const { dataFetched, loading, dataLoading, dataError } = this.props;

    return (
      <div style={SVGViewerStyles.svgContainer}>
        {!dataFetched && !loading && this.props.uploadHistory.length === 0 && (
          <div style={SVGViewerStyles.placeholderMessage}>
            <p>Upload one or more X-ray images to begin analysis</p>
            <p style={SVGViewerStyles.placeholderSubtext}>
              The images will appear here
            </p>
          </div>
        )}

        <svg
          ref={this.svgRef}
          style={{
            ...SVGViewerStyles.svg,
            ...(dataFetched ? SVGViewerStyles.svgWithData : {}),
          }}
        />

        {dataLoading && dataFetched && (
          <div style={SVGViewerStyles.loadingOverlay}>Loading image...</div>
        )}

        {dataError && !loading && (
          <div style={SVGViewerStyles.errorOverlay}>
            Error: {dataError.message}
          </div>
        )}
      </div>
    );
  }
}
