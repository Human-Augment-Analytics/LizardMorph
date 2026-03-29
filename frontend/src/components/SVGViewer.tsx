import { Component, createRef } from "react";
import * as d3 from "d3";
import { getSVGViewerStyles } from "./SVGViewer.style";
import type { ResolvedTheme } from "../contexts/ThemeContext";
import type { Point } from "../models/Point";
import type { UploadHistoryItem } from "../models/UploadHistoryItem";
import type { BoundingBox } from "../models/AnnotationsData";

interface SVGViewerProps {
  selectedViewType: string;
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
  theme: ResolvedTheme;
}

interface SVGViewerState {
  hintDismissed: boolean;
  landmarkSize: number;
  labelSize: number;
  labelColor: string;
  showLabels: boolean;
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

  /** Set in render() so the d3 zoom filter matches edit mode even when renderSVG does not re-run */
  private blockZoomWhileEditing = false;

  constructor(props: SVGViewerProps) {
    super(props);
    this.state = {
      hintDismissed: false,
      landmarkSize: props.selectedViewType === 'toepads' ? 1.0 : 2.0,
      labelSize: 8,
      labelColor: '#ffffff',
      showLabels: true,
      showBoundingBoxes: false,
      showControls: false
    };
  }
  
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
      (prevProps.imageWidth !== this.props.imageWidth ||
        prevProps.imageHeight !== this.props.imageHeight ||
        prevProps.maxWidth !== this.props.maxWidth ||
        this.props.needsScaling ||
        prevProps.boundingBoxes !== this.props.boundingBoxes ||
        (this.props.boundingBoxes && this.props.boundingBoxes.length !== (prevProps.boundingBoxes?.length || 0)) ||
        prevProps.originalScatterData.length !== this.props.originalScatterData.length)
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

      // Re-sync d3 zoom internal transform when leaving edit mode (zoom may have been
      // skipped on first paint in edit mode, or filter must match props before pan/zoom)
      if (
        !this.props.isEditMode &&
        this.svgRef.current &&
        this.zoomRef.current
      ) {
        const svg = d3.select(this.svgRef.current);
        svg.call(this.zoomRef.current.transform, this.props.zoomTransform);
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
      // Set both src (for HTML img) and href (fallback for SVG image)
      imageElement.attr("src", this.props.currentImageURL);
      imageElement.attr("href", this.props.currentImageURL);
      imageElement.attr("xlink:href", this.props.currentImageURL);
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
      this.props.imageHeight
    ) {
      console.log(
        "Rendering SVG with image dimensions:",
        this.props.imageWidth,
        "x",
        this.props.imageHeight
      );

      const svg = d3.select<SVGSVGElement, unknown>(this.svgRef.current!);
      svg.selectAll("*").remove(); // Clear SVG first to prevent duplication

      // Add drop shadow filter for label readability (no stroke outline needed)
      const defs = svg.append("defs");
      const filter = defs.append("filter")
        .attr("id", "label-shadow")
        .attr("x", "-20%").attr("y", "-20%")
        .attr("width", "140%").attr("height", "140%");
      filter.append("feDropShadow")
        .attr("dx", "0").attr("dy", "0")
        .attr("stdDeviation", "1.5")
        .attr("flood-color", "black")
        .attr("flood-opacity", "0.9");

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
          .domain([0.5, this.props.imageWidth + 0.5])
          .range([0, width]);

        const scaleY = d3
          .scaleLinear()
          .domain([0.5, this.props.imageHeight + 0.5])
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
      // Add image to the zoom container using foreignObject to bypass SVG-specific rendering bugs and security sandboxes
      zoomContainer
        .append("foreignObject")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", width)
        .attr("height", height)
        .append("xhtml:img")
        .attr("class", "background-img")
        .attr("src", this.props.currentImageURL)
        .style("width", "100%")
        .style("height", "100%")
        .style("object-fit", "fill")
        .style("pointer-events", "none")
        .attr("draggable", "false");

      // Add bounding boxes to the zoom container (if available and visible)
      if (this.state.showBoundingBoxes && this.props.boundingBoxes && this.props.boundingBoxes.length > 0) {
        const bboxGroup = zoomContainer
          .append("g")
          .attr("class", "bounding-boxes");
        
        // Scale functions for bounding boxes
        const scaleX = d3.scaleLinear()
          .domain([0.5, this.props.imageWidth + 0.5])
          .range([0, width]);
        const scaleY = d3.scaleLinear()
          .domain([0.5, this.props.imageHeight + 0.5])
          .range([0, height]);
        
        // Color scheme based on class label
        const labelColors: Record<string, string> = {
          'bot_finger': '#00ff00', 'up_finger': '#00cc00', 'finger': '#00ff00', 'toe/finger': '#00ff00',
          'bot_toe': '#0088ff', 'up_toe': '#0066dd', 'toe': '#0088ff',
          'ruler': '#ff8800', 'scale': '#ff8800',
          'id': '#cc44ff',
        };
        const defaultColor = '#ffffff';
        
        this.props.boundingBoxes.forEach((bbox, index) => {
          const scaledX = scaleX(bbox.left);
          const scaledY = scaleY(bbox.top);
          const scaledWidth = scaleX(bbox.width);
          const scaledHeight = scaleY(bbox.height);
          
          const label = (bbox.label || '').toLowerCase();
          const color = labelColors[label] || defaultColor;
          
          // Calculate coverage for logging
          const boxArea = bbox.width * bbox.height;
          const imageArea = this.props.imageWidth * this.props.imageHeight;
          const coverageRatio = boxArea / imageArea;
          
          console.log(`Rendering bounding box ${index}:`, {
            original: { left: bbox.left, top: bbox.top, width: bbox.width, height: bbox.height },
            scaled: { x: scaledX, y: scaledY, width: scaledWidth, height: scaledHeight },
            coverage: `${(coverageRatio * 100).toFixed(1)}%`,
            label: bbox.label || 'unknown'
          });
          
          if (bbox.obb_corners && bbox.obb_corners.length === 4) {
            // Draw rotated bounding box via polygon
            const pointsString = bbox.obb_corners.map(p => `${scaleX(p.x)},${scaleY(p.y)}`).join(" ");
            bboxGroup
              .append("polygon")
              .attr("points", pointsString)
              .attr("fill", "none")
              .attr("stroke", color)
              .attr("stroke-width", 3)
              .attr("stroke-dasharray", "8,4")
              .attr("opacity", 0.9)
              .style("pointer-events", "none");
            
            // Add label text near the first corner of the OBB
            if (bbox.label) {
              const firstCorner = bbox.obb_corners[0];
              bboxGroup
                .append("text")
                .attr("x", scaleX(firstCorner.x))
                .attr("y", scaleY(firstCorner.y) - 4)
                .text(bbox.label)
                .attr("fill", color)
                .attr("font-size", "11px")
                .attr("font-weight", "bold")
                .attr("stroke", "black")
                .attr("stroke-width", "0.5px")
                .style("pointer-events", "none");
            }
          } else {
            // Draw axis-aligned bounding box
            bboxGroup
              .append("rect")
              .attr("x", scaledX)
              .attr("y", scaledY)
              .attr("width", scaledWidth)
              .attr("height", scaledHeight)
              .attr("fill", "none")
              .attr("stroke", color)
              .attr("stroke-width", 3)
              .attr("stroke-dasharray", "8,4")
              .attr("opacity", 0.9)
              .style("pointer-events", "none");
            
            // Add label text above the box
            if (bbox.label) {
              bboxGroup
                .append("text")
                .attr("x", scaledX)
                .attr("y", scaledY - 4)
                .text(bbox.label)
                .attr("fill", color)
                .attr("font-size", "11px")
                .attr("font-weight", "bold")
                .attr("stroke", "black")
                .attr("stroke-width", "0.5px")
                .style("pointer-events", "none");
            }
          }
        });
      }

      // Add scatter points to the zoom container
      const scatterPlotGroup = zoomContainer
        .append("g")
        .attr("class", "scatter-points");

      // Always scale from originalScatterData to match current SVG dimensions
      // (this ensures correct scaling when maxWidth constraint is applied)
      const scaleXForPoints = d3.scaleLinear()
        .domain([0.5, this.props.imageWidth + 0.5])
        .range([0, width]);
      const scaleYForPoints = d3.scaleLinear()
        .domain([0.5, this.props.imageHeight + 0.5])
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
        // const fontSize = Math.max(6, size * 2);
        // const textOffset = size + 1;
        
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
        if (this.state.showLabels) {
          const textOffset = size + 1;
          g.append("text")
            .attr("x", d.x + textOffset)
            .attr("y", d.y - textOffset)
            .text(d.id + 1)
            .attr("font-size", `${this.state.labelSize}px`)
            .attr("fill", this.state.labelColor)
            .attr("stroke", "none")
            .attr("filter", "url(#label-shadow)")
            .attr("opacity", this.isTransparentMode ? 0.6 : 1.0)
            .style("pointer-events", "none");
        }
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
        .scaleExtent([0.5, 20])
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
          // Disable zoom/pan while editing (read live flag from render(), not props at renderSVG time)
          if (this.blockZoomWhileEditing) {
            return false;
          }
          return !mouseEvent.button && event.type !== "dblclick";
        });

      // Store the zoom reference for external control
      this.zoomRef.current = zoom;

      // Always attach zoom; filter + blockZoomWhileEditing block interaction in edit mode
      svg.call(zoom);

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

      // Click-to-place: in free mode + edit mode, clicking the background adds a new landmark
      if (this.props.selectedViewType === "free") {
        svg.on("click.place", (event: MouseEvent) => {
          if (!this.props.isEditMode) return;
          // Ignore clicks on existing landmarks (circles/text)
          const target = event.target as Element;
          if (target.tagName === "circle" || target.tagName === "text") return;
          // Also ignore if the click is on a landmark group child
          const parentTag = target.parentElement?.tagName;
          if (parentTag === "g" && target.parentElement?.classList.contains("landmark-group")) return;

          this.updateCachedScales();
          if (!this.cachedScales.scaleXToImg || !this.cachedScales.scaleYToImg) return;

          // Get click position in SVG coordinates, accounting for zoom transform
          const pt = d3.pointer(event, this.svgRef.current!);
          const transform = d3.zoomTransform(this.svgRef.current!);
          const svgX = (pt[0] - transform.x) / transform.k;
          const svgY = (pt[1] - transform.y) / transform.k;

          // Convert to image coordinates
          const imgX = this.cachedScales.scaleXToImg(svgX);
          const imgY = this.cachedScales.scaleYToImg(svgY);

          // Generate next sequential ID
          const maxId = this.props.originalScatterData.length > 0
            ? Math.max(...this.props.originalScatterData.map(p => p.id))
            : -1;
          const newId = maxId + 1;

          // Save history for undo
          this.saveToHistory();

          // Create new point in image space
          const newOriginalPoint: Point = { id: newId, x: imgX, y: imgY };

          // Create display-space version
          const newDisplayPoint: Point = {
            id: newId,
            x: this.cachedScales.scaleXDisplay ? this.cachedScales.scaleXDisplay(imgX) : svgX,
            y: this.cachedScales.scaleYDisplay ? this.cachedScales.scaleYDisplay(imgY) : svgY,
          };

          // Update data arrays
          const updatedOriginal = [...this.props.originalScatterData, newOriginalPoint];
          const updatedDisplay = [...this.props.scatterData, newDisplayPoint];

          // Propagate to parent — componentDidUpdate detects length change and re-renders
          this.props.onScatterDataUpdate(updatedDisplay, updatedOriginal);
        });
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
    // Raise for z-order only — do not set stroke on <g>; it inherits to <text> and outlines the numbers
    clickedGroup.raise();
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
    if (!this.isDragging || this.draggedLandmarkId === null || !this.dragStartPosition) return;
    
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
    // Clear any inherited stroke on the group (legacy drags used stroke here and outlined labels)
    d3.select(event.sourceEvent.target.parentNode as Element).attr("stroke", "none");
    
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
      const pointCountChanged = lastState.originalScatterData.length !== this.props.originalScatterData.length;
      this.props.onScatterDataUpdate(lastState.scatterData, lastState.originalScatterData);
      if (pointCountChanged) {
        // Point was added or removed — componentDidUpdate will trigger full re-render
        // via the length check
      } else {
        // Only positions changed — update in place
        this.updateSVGPositions(lastState.scatterData);
      }
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
    const textOffset = size + 1;

    // Update circle sizes
    scatterPlotGroup.selectAll<SVGCircleElement, Point>("circle")
      .attr("r", size);

    // Update text sizes, positions, and color
    scatterPlotGroup.selectAll<SVGTextElement, Point>("text")
      .attr("font-size", `${this.state.labelSize}px`)
      .attr("fill", this.state.labelColor)
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

  // Handle label size change
  private handleLabelSizeChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
    const newSize = parseFloat(event.target.value);
    this.setState({ labelSize: newSize }, () => {
      this.updateLandmarkSizes();
    });
  };

  // Handle label color change
  private handleLabelColorChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
    this.setState({ labelColor: event.target.value }, () => {
      this.updateLandmarkSizes();
    });
  };

  // Delete the currently selected landmark (free mode)
  private deleteSelectedLandmark = (): void => {
    if (!this.props.selectedPoint) return;
    const idToDelete = this.props.selectedPoint.id;

    // Save history for undo
    this.saveToHistory();

    // Filter out the deleted point
    const updatedOriginal = this.props.originalScatterData.filter(p => p.id !== idToDelete);
    const updatedDisplay = this.props.scatterData.filter(p => p.id !== idToDelete);

    // Clear selection and update data — componentDidUpdate detects length change and re-renders
    this.props.onPointSelect(null);
    this.props.onScatterDataUpdate(updatedDisplay, updatedOriginal);
  };

  // Toggle label visibility
  private toggleLabels = (): void => {
    this.setState({ showLabels: !this.state.showLabels }, () => {
      this.renderSVG();
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

    // Toggle labels with 'N' key
    if (event.key.toLowerCase() === 'n' && !event.ctrlKey && !event.metaKey) {
      event.preventDefault();
      this.toggleLabels();
    }

    // Delete selected landmark with Delete/Backspace (free mode)
    if ((event.key === 'Delete' || event.key === 'Backspace') && !event.ctrlKey && !event.metaKey) {
      if (this.props.selectedPoint && this.props.selectedViewType === "free") {
        event.preventDefault();
        this.deleteSelectedLandmark();
      }
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
      
      this.cachedScales.scaleXToImg = d3.scaleLinear().domain([0, width]).range([0.5, this.props.imageWidth + 0.5]);
      this.cachedScales.scaleYToImg = d3.scaleLinear().domain([0, height]).range([0.5, this.props.imageHeight + 0.5]);
      this.cachedScales.scaleXDisplay = d3.scaleLinear().domain([0.5, this.props.imageWidth + 0.5]).range([0, width]);
      this.cachedScales.scaleYDisplay = d3.scaleLinear().domain([0.5, this.props.imageHeight + 0.5]).range([0, height]);
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
    this.blockZoomWhileEditing = this.props.isEditMode;
    const { dataFetched, loading, dataLoading, dataError } = this.props;
    const viewerStyles = getSVGViewerStyles(this.props.theme);

    return (
      <div style={viewerStyles.svgContainer}>
        {!dataFetched && !loading && this.props.uploadHistory.length === 0 && (
          <div style={viewerStyles.placeholderMessage}>
            <p>Upload one or more X-ray images to begin analysis</p>
            <p style={viewerStyles.placeholderSubtext}>
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
            <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
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
            </div>

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

                {/* Label Size slider */}
                <div style={{ marginBottom: '12px' }}>
                  <div style={{ marginBottom: '6px', fontWeight: 'bold', fontSize: '11px', opacity: 0.9, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>Label Size</span>
                    <button
                      onClick={this.toggleLabels}
                      style={{
                        padding: '2px 8px',
                        fontSize: '10px',
                        background: this.state.showLabels ? '#4CAF50' : '#666',
                        color: 'white',
                        border: 'none',
                        borderRadius: '3px',
                        cursor: 'pointer',
                      }}
                    >
                      {this.state.showLabels ? 'ON' : 'OFF'}
                    </button>
                  </div>
                  <input
                    type="range"
                    min="4"
                    max="24"
                    step="1"
                    value={this.state.labelSize}
                    onChange={this.handleLabelSizeChange}
                    style={{
                      width: '100%',
                      marginBottom: '2px'
                    }}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '10px', opacity: 0.7 }}>
                    <span>{this.state.labelSize}px</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <span>Color</span>
                      <input
                        type="color"
                        value={this.state.labelColor}
                        onChange={this.handleLabelColorChange}
                        style={{ width: '20px', height: '16px', border: 'none', cursor: 'pointer', background: 'none' }}
                      />
                    </div>
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
                  <div><kbd style={{ background: '#444', padding: '1px 4px', borderRadius: '2px', marginRight: '6px' }}>N</kbd> Toggle labels</div>
                  <div><kbd style={{ background: '#444', padding: '1px 4px', borderRadius: '2px', marginRight: '6px' }}>O</kbd> Toggle outline mode</div>
                  <div><kbd style={{ background: '#444', padding: '1px 4px', borderRadius: '2px', marginRight: '6px' }}>Right-click</kbd> Edit/view mode</div>
                  {this.props.selectedViewType === "free" && (
                    <div><kbd style={{ background: '#444', padding: '1px 4px', borderRadius: '2px', marginRight: '6px' }}>Delete</kbd> Remove selected point</div>
                  )}
                  <div><kbd style={{ background: '#444', padding: '1px 4px', borderRadius: '2px', marginRight: '6px' }}>Ctrl+Z</kbd> Undo</div>
                </div>
              </div>
            )}
          </div>
        )}

        <svg
          ref={this.svgRef}
          xmlns="http://www.w3.org/2000/svg"
          xmlnsXlink="http://www.w3.org/1999/xlink"
          style={{
            ...viewerStyles.svg,
            ...(dataFetched ? viewerStyles.svgWithData : {}),
          }}
          onContextMenu={this.handleContextMenu}
        />

        {dataLoading && dataFetched && (
          <div style={viewerStyles.loadingOverlay}>Loading image...</div>
        )}

        {dataError && !loading && (
          <div style={viewerStyles.errorOverlay}>
            Error: {dataError.message}
          </div>
        )}
      </div>
    );
  }
}