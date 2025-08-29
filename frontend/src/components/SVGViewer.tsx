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
  onPointSelect: (point: Point | null) => void;
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
  
  // Performance optimization: Cache scale functions and dimensions
  private cachedScales: {
    scaleXToImg: d3.ScaleLinear<number, number> | null;
    scaleYToImg: d3.ScaleLinear<number, number> | null;
    scaleXDisplay: d3.ScaleLinear<number, number> | null;
    scaleYDisplay: d3.ScaleLinear<number, number> | null;
    svgWidth: number;
    svgHeight: number;
  } = {
    scaleXToImg: null,
    scaleYToImg: null,
    scaleXDisplay: null,
    scaleYDisplay: null,
    svgWidth: 0,
    svgHeight: 0,
  };

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

  // Handle right-click to toggle edit mode
  private handleContextMenu = (event: React.MouseEvent): void => {
    event.preventDefault(); // Prevent default context menu
    this.props.onToggleEditMode();
  };

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
        .append("g")
        .attr("data-landmark-id", (d: Point) => d.id) // Add unique identifier to each group
        .attr("class", "landmark-group");

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
          .attr("stroke-width", "0.5px")
          .style("pointer-events", "none"); // Prevent text from interfering with drag
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
          return this.props.selectedPoint && d.id === this.props.selectedPoint.id
            ? "yellow"
            : "red";
        })
        .attr("stroke-width", (d: Point) => {
          return this.props.selectedPoint && d.id === this.props.selectedPoint.id
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
    
    // Only allow drag if this landmark is already selected (yellow) or if no landmark is currently selected
    const isCurrentlySelected = this.props.selectedPoint && this.props.selectedPoint.id === d.id;
    const noLandmarkSelected = !this.props.selectedPoint;
    
    if (!isCurrentlySelected && !noLandmarkSelected) {
      // If clicking on a non-selected landmark while another is selected, just select it without starting drag
      this.props.onPointSelect(d);
      
      // Update visual selection
      const svg = d3.select(this.svgRef.current);
      const scatterPlotGroup = svg.select<SVGGElement>(".scatter-points");
      scatterPlotGroup.selectAll<SVGGElement, Point>("g").each((pointData, i, nodes) => {
        const group = d3.select(nodes[i]);
        const circle = group.select("circle");
        if (pointData.id === d.id) {
          circle.attr("fill", "yellow").attr("stroke-width", 2);
        } else {
          circle.attr("fill", "red").attr("stroke-width", 1);
        }
      });
      return; // Don't start dragging
    }
    
    this.isDragging = true;
    this.draggedLandmarkId = d.id; // Store the landmark ID being dragged
    
    // Use the data directly from the drag event instead of DOM traversal
    const clickedData = d;
    
    // Find the group element for this landmark
    const clickedGroup = d3.select(event.sourceEvent.target.parentNode as Element);
    clickedGroup.raise().attr("stroke", "black");
    this.props.onPointSelect(clickedData);
    
    // Update visual selection by finding the correct group and updating its circle
    const svg = d3.select(this.svgRef.current);
    const scatterPlotGroup = svg.select<SVGGElement>(".scatter-points");
    scatterPlotGroup.selectAll<SVGGElement, Point>("g").each((pointData, i, nodes) => {
      const group = d3.select(nodes[i]);
      const circle = group.select("circle");
      if (pointData.id === clickedData.id) {
        circle.attr("fill", "yellow").attr("stroke-width", 2);
      } else {
        circle.attr("fill", "red").attr("stroke-width", 1);
      }
    });
  };

  private readonly dragged = (
    event: d3.D3DragEvent<SVGGElement, Point, Point>
  ): void => {
    // Only allow dragging if we're actually dragging (isDragging is true)
    if (!this.isDragging || !this.draggedLandmarkId) return;
    
    // Performance optimization: Update cached scales if needed
    this.updateCachedScales();
    
    // Use the stored landmark ID to ensure we're dragging the correct landmark
    const clickedData = this.getLandmarkData(this.draggedLandmarkId);
    if (!clickedData) return;
    
    // Find the group element for this landmark
    const clickedGroup = d3.select(this.svgRef.current)
      .select(`g[data-landmark-id="${this.draggedLandmarkId}"]`);
    
    const point = d3.pointer(event, this.svgRef.current);
    const transform = d3.zoomTransform(this.svgRef.current!);

    // SVG/display coordinates
    const x = (point[0] - transform.x) / transform.k;
    const y = (point[1] - transform.y) / transform.k;

    // Use cached scale functions for better performance
    if (this.cachedScales.scaleXToImg && this.cachedScales.scaleYToImg) {
      const newImgX = this.cachedScales.scaleXToImg(x);
      const newImgY = this.cachedScales.scaleYToImg(y);

      // Performance optimization: Only update the visual position immediately
      // Don't update parent component on every drag event
      clickedGroup.select("circle").attr("cx", x).attr("cy", y);
      clickedGroup.select("text").attr("x", x + 5).attr("y", y - 5);

      // Performance optimization: Throttle parent component updates
      if (this.dragUpdateTimeout) {
        clearTimeout(this.dragUpdateTimeout);
      }

      this.dragUpdateTimeout = window.setTimeout(() => {
        if (!this.isDragging) return;
        
        // Update originalScatterData (image space)
        const updatedoriginalScatterData = this.props.originalScatterData.map(
          (p: Point) =>
            p.id === clickedData.id
              ? { ...p, x: newImgX, y: newImgY }
              : p
        );

        // Re-scale for display using cached functions
        if (this.cachedScales.scaleXDisplay && this.cachedScales.scaleYDisplay) {
          const updatedScatterData = updatedoriginalScatterData.map(
            (p: Point) => ({
              ...p,
              x: this.cachedScales.scaleXDisplay!(p.x),
              y: this.cachedScales.scaleYDisplay!(p.y),
            })
          );

          // Update the display and image-space coordinates in parent
          this.props.onScatterDataUpdate(updatedScatterData, updatedoriginalScatterData);
        }
      }, 16); // ~60fps throttling
    }
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

      this.props.onPointSelect(null);
    }
  };

  // Performance optimization: Throttle drag updates
  private dragUpdateTimeout: number | null = null;
  private isDragging = false;
  private draggedLandmarkId: number | null = null; // Track which landmark is being dragged

  // Performance optimization: Update cached scales when dimensions change
  private updateCachedScales = (): void => {
    if (!this.svgRef.current) return;
    
    const svg = d3.select(this.svgRef.current);
    const width = +svg.attr("width");
    const height = +svg.attr("height");
    
    // Only update if dimensions changed
    if (width !== this.cachedScales.svgWidth || height !== this.cachedScales.svgHeight) {
      this.cachedScales.svgWidth = width;
      this.cachedScales.svgHeight = height;
      
      this.cachedScales.scaleXToImg = d3.scaleLinear().domain([0, width]).range([0, this.props.imageWidth]);
      this.cachedScales.scaleYToImg = d3.scaleLinear().domain([0, height]).range([0, this.props.imageHeight]);
      this.cachedScales.scaleXDisplay = d3.scaleLinear().domain([0, this.props.imageWidth]).range([0, width]);
      this.cachedScales.scaleYDisplay = d3.scaleLinear().domain([0, this.props.imageHeight]).range([0, height]);
    }
  };

  // Helper function to get landmark data by ID
  private getLandmarkData = (landmarkId: number): Point | null => {
    return this.props.scatterData.find(point => point.id === landmarkId) || null;
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

        {/* Mode indicator */}
        {dataFetched && (
          <div style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            background: this.props.isEditMode ? '#ff6b6b' : '#4ecdc4',
            color: 'white',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '12px',
            fontWeight: 'bold',
            zIndex: 1000,
            boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
          }}>
            {this.props.isEditMode ? 'EDIT MODE' : 'VIEW MODE'}
          </div>
        )}

        {/* Right-click hint */}
        {dataFetched && (
          <div style={{
            position: 'absolute',
            bottom: '10px',
            left: '10px',
            background: 'rgba(0,0,0,0.7)',
            color: 'white',
            padding: '6px 10px',
            borderRadius: '4px',
            fontSize: '11px',
            zIndex: 1000,
            opacity: 0.8
          }}>
            Right-click to toggle edit/view mode
          </div>
        )}

        <svg
          ref={this.svgRef}
          style={{
            ...SVGViewerStyles.svg,
            ...(dataFetched ? SVGViewerStyles.svgWithData : {}),
          }}
          onContextMenu={this.handleContextMenu}
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
