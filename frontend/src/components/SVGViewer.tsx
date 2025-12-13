import { Component, createRef } from "react";
import * as d3 from "d3";
import { SVGViewerStyles } from "./SVGViewer.style";
import type { Point } from "../models/Point";
import type { UploadHistoryItem } from "../models/UploadHistoryItem";
import type { BoundingBox } from "../models/AnnotationsData";

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
  isModalOpen?: boolean;
  boundingBoxes?: BoundingBox[];
  maxWidth?: number; // Optional max width for constrained views like toepads
  fitToContainerWidth?: boolean; // Scale to fit container width instead of height
}

interface SVGViewerState {
  hintDismissed: boolean;
  landmarkSize: number;
  showBoundingBoxes: boolean;
  showControls: boolean;
}

interface PositionHistory {
  scatterData: Point[];
  originalScatterData: Point[];
}

export class SVGViewer extends Component<SVGViewerProps, SVGViewerState> {
  readonly svgRef = createRef<SVGSVGElement>();
  readonly zoomRef = createRef<d3.ZoomBehavior<SVGSVGElement, unknown>>();
  
  state: SVGViewerState = {
    hintDismissed: false,
    landmarkSize: 2,
    showBoundingBoxes: false,
    showControls: false
  };
  
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

  componentDidMount() {
    // Add keyboard event listener
    window.addEventListener('keydown', this.handleKeyDown);
  }

  componentWillUnmount() {
    // Remove keyboard event listener
    window.removeEventListener('keydown', this.handleKeyDown);
  }

  componentDidUpdate(prevProps: SVGViewerProps) {
    // Handle image URL changes without full re-render to preserve zoom
    if (prevProps.currentImageURL !== this.props.currentImageURL && 
        this.props.currentImageURL && 
        this.svgRef.current) {
      this.updateImageSource();
      return;
    }

    if (
      this.props.currentImageURL &&
      this.props.imageWidth &&
      this.props.imageHeight &&
      this.props.originalScatterData.length > 0 &&
      (prevProps.imageWidth !== this.props.imageWidth ||
        prevProps.imageHeight !== this.props.imageHeight ||
        prevProps.maxWidth !== this.props.maxWidth ||
        this.props.needsScaling ||
        prevProps.boundingBoxes !== this.props.boundingBoxes ||
        (this.props.boundingBoxes && this.props.boundingBoxes.length !== (prevProps.boundingBoxes?.length || 0)))
    ) {
      setTimeout(() => {
        this.renderSVG();
      }, 0);
    }

    // Update drag behavior if edit mode changed
    if (prevProps.isEditMode !== this.props.isEditMode) {
      this.updateDragBehavior();
      
      // Reset history when edit mode starts on a new image
      if (this.props.isEditMode && this.currentImageId !== this.props.currentImageURL) {
        this.resetHistory();
      }
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

  // Update only the image source without re-rendering the entire SVG
  private updateImageSource = (): void => {
    if (!this.svgRef.current) return;
    
    const svg = d3.select(this.svgRef.current);
    const imageElement = svg.select(".background-img");
    
    if (!imageElement.empty()) {
      imageElement.attr("href", this.props.currentImageURL);
    }
    
    // Apply the stored zoom transform to preserve zoom state in edit mode
    if (this.props.zoomTransform && this.props.zoomTransform !== d3.zoomIdentity) {
      const zoomContainer = svg.select(".zoom-container");
      if (!zoomContainer.empty()) {
        const transformString = this.props.zoomTransform.toString();
        zoomContainer.attr("transform", transformString);
      }
    }
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
      let width: number;
      let height: number;

      if (this.props.fitToContainerWidth) {
        // Scale based on container width (use available window width minus sidebar)
        const containerWidth = window.innerWidth * 0.65; // Approximate available width
        width = containerWidth;
        height = width * (this.props.imageHeight / this.props.imageWidth);

        // Cap height to window height if too tall
        const maxHeight = window.innerHeight * 0.7;
        if (height > maxHeight) {
          height = maxHeight;
          width = height * (this.props.imageWidth / this.props.imageHeight);
        }
      } else {
      // Default: scale based on window height
        const windowHeight = window.innerHeight - window.innerHeight * 0.2;
        width = windowHeight * (this.props.imageWidth / this.props.imageHeight);
        height = windowHeight;

        // If maxWidth is provided, constrain to it
        if (this.props.maxWidth && width > this.props.maxWidth) {
          width = this.props.maxWidth;
          height = width * (this.props.imageHeight / this.props.imageWidth);
        }
      }

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

      // Add bounding boxes to the zoom container (if available and visible)
      if (this.state.showBoundingBoxes && this.props.boundingBoxes && this.props.boundingBoxes.length > 0) {
        const bboxGroup = zoomContainer
          .append("g")
          .attr("class", "bounding-boxes");
        
        // Scale functions for bounding boxes
        const scaleX = d3.scaleLinear()
          .domain([0, this.props.imageWidth])
          .range([0, width]);
        const scaleY = d3.scaleLinear()
          .domain([0, this.props.imageHeight])
          .range([0, height]);
        
        // Color scheme for different box types (toe, finger, scale)
        const boxColors = ['#00ff00', '#0088ff', '#ff8800']; // Green, Blue, Orange
        
        this.props.boundingBoxes.forEach((bbox, index) => {
          const scaledX = scaleX(bbox.left);
          const scaledY = scaleY(bbox.top);
          const scaledWidth = scaleX(bbox.width);
          const scaledHeight = scaleY(bbox.height);
          
          // Calculate coverage for logging
          const boxArea = bbox.width * bbox.height;
          const imageArea = this.props.imageWidth * this.props.imageHeight;
          const coverageRatio = boxArea / imageArea;
          
          console.log(`Rendering bounding box ${index}:`, {
            original: { left: bbox.left, top: bbox.top, width: bbox.width, height: bbox.height },
            scaled: { x: scaledX, y: scaledY, width: scaledWidth, height: scaledHeight },
            coverage: `${(coverageRatio * 100).toFixed(1)}%`
          });
          
          // Draw rectangle for bounding box
          bboxGroup
            .append("rect")
            .attr("x", scaledX)
            .attr("y", scaledY)
            .attr("width", scaledWidth)
            .attr("height", scaledHeight)
            .attr("fill", "none")
            .attr("stroke", boxColors[index % boxColors.length])
            .attr("stroke-width", 3)
            .attr("stroke-dasharray", "8,4")
            .attr("opacity", 0.9)
            .style("pointer-events", "none");
        });
      }

      // Add scatter points to the zoom container
      const scatterPlotGroup = zoomContainer
        .append("g")
        .attr("class", "scatter-points");

      // Always scale from originalScatterData to match current SVG dimensions
      // (this ensures correct scaling when maxWidth constraint is applied)
      const scaleXForPoints = d3.scaleLinear()
        .domain([0, this.props.imageWidth])
        .range([0, width]);
      const scaleYForPoints = d3.scaleLinear()
        .domain([0, this.props.imageHeight])
        .range([0, height]);

      const scaledPointsForRender = this.props.originalScatterData.map((point: Point) => ({
        ...point,
        x: scaleXForPoints(point.x),
        y: scaleYForPoints(point.y),
      }));

      const pointGroups = scatterPlotGroup
        .selectAll("g")
        .data(scaledPointsForRender)
        .enter()
        .append("g")
        .attr("data-landmark-id", (d: Point) => d.id) // Add unique identifier to each group
        .attr("class", "landmark-group");

      pointGroups.each((d, i, nodes) => {
        const g = d3.select(nodes[i]);

        // Add the point
        const size = this.state.landmarkSize;
        const fontSize = Math.max(6, size * 2);
        const textOffset = size + 1;
        
        g.append("circle")
          .attr("cx", d.x)
          .attr("cy", d.y)
          .attr("r", size)
          .attr(
            "fill",
            this.isOutlineMode ? "transparent" : (
              this.props.selectedPoint && d.id === this.props.selectedPoint.id
                ? "yellow"
                : "red"
            )
          )
          .attr("stroke", this.isOutlineMode ? (
            this.props.selectedPoint && d.id === this.props.selectedPoint.id
              ? "yellow"
              : "red"
          ) : "none")
          .attr("stroke-width", this.isOutlineMode ? "0.5" : "0")
          .attr("data-id", d.id)
          .attr("opacity", this.isTransparentMode ? 0.3 : 1.0)
          .style("cursor", "pointer")
          .style("pointer-events", "all");

        // Add the number label
        g.append("text")
          .attr("x", d.x + textOffset)
          .attr("y", d.y - textOffset)
          .text(d.id)
          .attr("font-size", `${fontSize}px`)
          .attr("fill", "white")
          .attr("stroke", "black")
          .attr("stroke-width", "0.5px")
          .attr("opacity", this.isTransparentMode ? 0.6 : 1.0)
          .style("pointer-events", "none"); // Prevent text from interfering with drag
      });

      // Set data-landmark-id on the groups for proper drag selection
      pointGroups.attr("data-landmark-id", (d: Point) => d.id);

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
        this.props.zoomTransform !== d3.zoomIdentity
      ) {
        if (!this.props.isEditMode) {
          // In view mode, use the zoom behavior to apply transform
          svg.call(zoom.transform, this.props.zoomTransform);
        } else {
          // In edit mode, directly apply transform to zoom container
          zoomContainer.attr("transform", this.props.zoomTransform.toString());
        }
      }
    }
  };

  private readonly updatePointSelection = (): void => {
    if (this.svgRef.current && this.props.scatterData.length > 0) {
      const svg = d3.select(this.svgRef.current);
      svg
        .selectAll<SVGCircleElement, Point>("circle")
        .attr("fill", (d: Point) => {
          if (this.isOutlineMode) {
            return "transparent";
          }
          return this.props.selectedPoint && d.id === this.props.selectedPoint.id
            ? "yellow"
            : "red";
        })
        .attr("stroke", (d: Point) => {
          if (this.isOutlineMode) {
            return this.props.selectedPoint && d.id === this.props.selectedPoint.id
              ? "yellow"
              : "red";
          }
          return "none";
        })
        .attr("stroke-width", this.isOutlineMode ? "0.5" : "0")
        .attr("opacity", this.isTransparentMode ? 0.3 : 1.0);
      
      // Update text opacity as well
      svg
        .selectAll<SVGTextElement, Point>("text")
        .attr("opacity", this.isTransparentMode ? 0.6 : 1.0);
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
      this.handlePointClick(d);
      
      // Update visual selection
      const svg = d3.select(this.svgRef.current);
      const scatterPlotGroup = svg.select<SVGGElement>(".scatter-points");
      scatterPlotGroup.selectAll<SVGGElement, Point>("g").each((pointData, i, nodes) => {
        const group = d3.select(nodes[i]);
        const circle = group.select("circle");
        if (pointData.id === d.id) {
          // Preserve outline mode when updating selection
          if (this.isOutlineMode) {
            circle.attr("fill", "transparent").attr("stroke", "yellow");
          } else {
            circle.attr("fill", "yellow").attr("stroke", "none");
          }
        } else {
          // Preserve outline mode when updating selection
          if (this.isOutlineMode) {
            circle.attr("fill", "transparent").attr("stroke", "red");
          } else {
            circle.attr("fill", "red").attr("stroke", "none");
          }
        }
      });
      return; // Don't start dragging
    }
    
    // Save current state to history before starting drag
    this.saveToHistory();
    
    // Store the starting position to distinguish between clicks and drags
    const point = d3.pointer(event, this.svgRef.current);
    this.dragStartPosition = { x: point[0], y: point[1] };
    
    this.isDragging = true;
    this.draggedLandmarkId = d.id; // Store the landmark ID being dragged
    
    // Use the data directly from the drag event instead of DOM traversal
    const clickedData = d;
    
    // Find the group element for this landmark
    const clickedGroup = d3.select(event.sourceEvent.target.parentNode as Element);
    clickedGroup.raise().attr("stroke", "black");
    this.handlePointClick(clickedData);
    
    // Update visual selection by finding the correct group and updating its circle
    const svg = d3.select(this.svgRef.current);
    const scatterPlotGroup = svg.select<SVGGElement>(".scatter-points");
    scatterPlotGroup.selectAll<SVGGElement, Point>("g").each((pointData, i, nodes) => {
      const group = d3.select(nodes[i]);
      const circle = group.select("circle");
      if (pointData.id === clickedData.id) {
        // Preserve outline mode when updating selection
        if (this.isOutlineMode) {
          circle.attr("fill", "transparent").attr("stroke", "yellow");
        } else {
          circle.attr("fill", "yellow").attr("stroke", "none");
        }
      } else {
        // Preserve outline mode when updating selection
        if (this.isOutlineMode) {
          circle.attr("fill", "transparent").attr("stroke", "red");
        } else {
          circle.attr("fill", "red").attr("stroke", "none");
        }
      }
    });
  };

  private readonly dragged = (
    event: d3.D3DragEvent<SVGGElement, Point, Point>
  ): void => {
    // Only allow dragging if we're actually dragging (isDragging is true)
    if (!this.isDragging || !this.draggedLandmarkId || !this.dragStartPosition) return;
    
    // Check if we've moved enough to consider this a drag (not just a click)
    const currentPoint = d3.pointer(event, this.svgRef.current);
    const distance = Math.sqrt(
      Math.pow(currentPoint[0] - this.dragStartPosition.x, 2) + 
      Math.pow(currentPoint[1] - this.dragStartPosition.y, 2)
    );
    
    if (distance < this.CLICK_THRESHOLD) {
      // Haven't moved enough to consider this a drag, just a click
      return;
    }
    
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
    
    // Reset drag state
    this.isDragging = false;
    this.draggedLandmarkId = null;
    this.dragStartPosition = null;
    
    // Clear any pending drag updates
    if (this.dragUpdateTimeout) {
      clearTimeout(this.dragUpdateTimeout);
      this.dragUpdateTimeout = null;
    }
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
  private dragStartPosition: { x: number; y: number } | null = null; // Track drag start position
  private readonly CLICK_THRESHOLD = 5; // Minimum pixel movement to consider as dragging

  // Transparency mode for landmarks
  private isTransparentMode = false;
  
  // Outline mode for landmarks
  private isOutlineMode = false;

  // Undo functionality - per image history
  private positionHistory: PositionHistory[] = [];
  private currentImageId: string | null = null;
  private readonly MAX_HISTORY_SIZE = 10;

  // Save current position state to history
  private saveToHistory = (): void => {
    const currentState: PositionHistory = {
      scatterData: [...this.props.scatterData],
      originalScatterData: [...this.props.originalScatterData]
    };
    
    this.positionHistory.push(currentState);
    
    // Limit history size
    if (this.positionHistory.length > this.MAX_HISTORY_SIZE) {
      this.positionHistory.shift();
    }
  };

  // Undo last position change
  private undoLastChange = (): void => {
    console.log("Attempting undo, history length:", this.positionHistory.length);
    if (this.positionHistory.length === 0) {
      return;
    }
    
    const lastState = this.positionHistory.pop();
    if (lastState) {
      this.props.onScatterDataUpdate(lastState.scatterData, lastState.originalScatterData);
      // Force visual update of SVG elements
      this.updateSVGPositions(lastState.scatterData);
    }
  };

  // Update SVG element positions directly
  private updateSVGPositions = (scatterData: Point[]): void => {
    if (!this.svgRef.current) return;
    
    const svg = d3.select(this.svgRef.current);
    const scatterPlotGroup = svg.select<SVGGElement>(".scatter-points");
    if (scatterPlotGroup.empty()) return;
    
    // Update circle positions
    scatterPlotGroup.selectAll<SVGCircleElement, Point>("circle")
      .data(scatterData, (d: Point) => d.id)
      .attr("cx", (d: Point) => d.x)
      .attr("cy", (d: Point) => d.y);
    
    // Update text positions
    scatterPlotGroup.selectAll<SVGTextElement, Point>("text")
      .data(scatterData, (d: Point) => d.id)
      .attr("x", (d: Point) => d.x + 5)
      .attr("y", (d: Point) => d.y - 5);
  };

  // Update landmark sizes
  private updateLandmarkSizes = (): void => {
    if (!this.svgRef.current) return;
    
    const svg = d3.select(this.svgRef.current);
    const scatterPlotGroup = svg.select<SVGGElement>(".scatter-points");
    if (scatterPlotGroup.empty()) return;
    
    const size = this.state.landmarkSize;
    const fontSize = Math.max(6, size * 2); // Scale font size with landmark size
    const textOffset = size + 1;
    
    // Update circle sizes
    scatterPlotGroup.selectAll<SVGCircleElement, Point>("circle")
      .attr("r", size);
    
    // Update text sizes and positions
    scatterPlotGroup.selectAll<SVGTextElement, Point>("text")
      .attr("font-size", `${fontSize}px`)
      .attr("x", (d: Point) => d.x + textOffset)
      .attr("y", (d: Point) => d.y - textOffset);
  };

  // Handle landmark size change
  private handleLandmarkSizeChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
    const newSize = parseFloat(event.target.value);
    this.setState({ landmarkSize: newSize }, () => {
      this.updateLandmarkSizes();
    });
  };

  // Reset history for new image
  private resetHistory = (): void => {
    this.positionHistory = [];
    this.currentImageId = this.props.currentImageURL;
  };

  // Toggle transparency mode
  private toggleTransparency = (): void => {
    this.isTransparentMode = !this.isTransparentMode;
    this.updateLandmarkTransparency();
  };

  // Toggle outline mode
  private toggleOutlineMode = (): void => {
    this.isOutlineMode = !this.isOutlineMode;
    this.updateLandmarkOutline();
  };

  // Toggle bounding boxes visibility
  private toggleBoundingBoxes = (): void => {
    this.setState({ showBoundingBoxes: !this.state.showBoundingBoxes }, () => {
      // Re-render SVG to show/hide bounding boxes
      this.renderSVG();
    });
  };

  // Toggle controls panel
  private toggleControls = (): void => {
    this.setState({ showControls: !this.state.showControls });
  };

  // Update landmark transparency
  private updateLandmarkTransparency = (): void => {
    if (!this.svgRef.current) return;
    
    const svg = d3.select(this.svgRef.current);
    const scatterPlotGroup = svg.select<SVGGElement>(".scatter-points");
    if (scatterPlotGroup.empty()) return;
    
    const opacity = this.isTransparentMode ? 0.3 : 1.0;
    
    // Update circles
    scatterPlotGroup.selectAll<SVGCircleElement, Point>("circle")
      .attr("opacity", opacity);
    
    // Update text (make it slightly more visible even in transparent mode)
    scatterPlotGroup.selectAll<SVGTextElement, Point>("text")
      .attr("opacity", this.isTransparentMode ? 0.6 : 1.0);
  };

  // Update landmark outline mode
  private updateLandmarkOutline = (): void => {
    if (!this.svgRef.current) return;
    
    const svg = d3.select(this.svgRef.current);
    const scatterPlotGroup = svg.select<SVGGElement>(".scatter-points");
    if (scatterPlotGroup.empty()) return;
    
    // Update circles to show only outline when in outline mode
    scatterPlotGroup.selectAll<SVGCircleElement, Point>("circle")
      .attr("fill", this.isOutlineMode ? "transparent" : (d: Point) => {
        return this.props.selectedPoint && d.id === this.props.selectedPoint.id
          ? "yellow"
          : "red";
      })
      .attr("stroke", this.isOutlineMode ? (d: Point) => {
        return this.props.selectedPoint && d.id === this.props.selectedPoint.id
          ? "yellow"
          : "red";
      } : "none")
      .attr("stroke-width", this.isOutlineMode ? "0.5" : "0");
  };

  // Handle keyboard events
  private handleKeyDown = (event: KeyboardEvent): void => {
    // Toggle transparency with 'T' key
    if (event.key.toLowerCase() === 't' && !event.ctrlKey && !event.metaKey) {
      event.preventDefault();
      this.toggleTransparency();
    }
    
    // Toggle outline mode with 'O' key
    if (event.key.toLowerCase() === 'o' && !event.ctrlKey && !event.metaKey) {
      event.preventDefault();
      this.toggleOutlineMode();
    }
    
    // Toggle bounding boxes with 'B' key
    if (event.key.toLowerCase() === 'b' && !event.ctrlKey && !event.metaKey) {
      event.preventDefault();
      this.toggleBoundingBoxes();
    }

    // Undo with Ctrl+Z
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
      event.preventDefault();
      this.undoLastChange();
    }
  };

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

  // Handle pure clicks (no dragging) to ensure proper point selection
  private handlePointClick = (point: Point): void => {
    // Only update selection if the point is different from currently selected
    if (!this.props.selectedPoint || this.props.selectedPoint.id !== point.id) {
      this.props.onPointSelect(point);
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

        {/* Info icon and expandable controls panel */}
        {dataFetched && (
          <div style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            zIndex: 1000,
          }}>
            {/* Info icon button */}
            <button
              onClick={this.toggleControls}
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '50%',
                background: this.state.showControls ? 'rgba(33, 150, 243, 0.9)' : 'rgba(0,0,0,0.7)',
                color: 'white',
                border: 'none',
                cursor: 'pointer',
                fontSize: '18px',
                fontWeight: 'bold',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'background 0.2s',
              }}
              title="Show/hide controls (keyboard shortcuts info)"
            >
              ℹ
            </button>

            {/* Expandable controls panel */}
            {this.state.showControls && (
              <div style={{
                marginTop: '8px',
                background: 'rgba(0,0,0,0.85)',
                color: 'white',
                padding: '12px',
                borderRadius: '8px',
                fontSize: '12px',
                minWidth: '220px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
              }}>
                {/* Landmark Size slider */}
                <div style={{ marginBottom: '12px' }}>
                  <div style={{ marginBottom: '6px', fontWeight: 'bold', fontSize: '11px', opacity: 0.9 }}>
                    Landmark Size
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="10"
                    step="0.5"
                    value={this.state.landmarkSize}
                    onChange={this.handleLandmarkSizeChange}
                    style={{
                      width: '100%',
                      marginBottom: '2px'
                    }}
                  />
                  <div style={{ textAlign: 'center', fontSize: '10px', opacity: 0.7 }}>
                    {this.state.landmarkSize.toFixed(1)}px
                  </div>
                </div>

                {/* Bounding box toggle */}
                <div style={{ 
                  marginBottom: '12px',
                  display: 'flex',
                  alignItems: 'center', 
                  justifyContent: 'space-between'
                }}>
                  <span style={{ fontSize: '11px' }}>Bounding Boxes</span>
                  <button
                    onClick={this.toggleBoundingBoxes}
                    style={{
                      padding: '4px 10px',
                      fontSize: '10px',
                      background: this.state.showBoundingBoxes ? '#4CAF50' : '#666',
                      color: 'white',
                      border: 'none',
                      borderRadius: '3px',
                      cursor: 'pointer',
                    }}
                  >
                    {this.state.showBoundingBoxes ? 'ON' : 'OFF'}
                  </button>
                </div>

                {/* Keyboard shortcuts */}
                <div style={{
                  borderTop: '1px solid rgba(255,255,255,0.2)',
                  paddingTop: '10px',
                  fontSize: '10px',
                  opacity: 0.8,
                  lineHeight: 1.6
                }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '4px', fontSize: '11px' }}>Keyboard Shortcuts</div>
                  <div><kbd style={{ background: '#444', padding: '1px 4px', borderRadius: '2px', marginRight: '6px' }}>B</kbd> Toggle bounding boxes</div>
                  <div><kbd style={{ background: '#444', padding: '1px 4px', borderRadius: '2px', marginRight: '6px' }}>T</kbd> Toggle transparency</div>
                  <div><kbd style={{ background: '#444', padding: '1px 4px', borderRadius: '2px', marginRight: '6px' }}>O</kbd> Toggle outline mode</div>
                  <div><kbd style={{ background: '#444', padding: '1px 4px', borderRadius: '2px', marginRight: '6px' }}>Right-click</kbd> Edit/view mode</div>
                  <div><kbd style={{ background: '#444', padding: '1px 4px', borderRadius: '2px', marginRight: '6px' }}>Ctrl+Z</kbd> Undo</div>
                </div>
              </div>
            )}
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