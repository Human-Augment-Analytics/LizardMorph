import React, { Component, createRef } from "react";
import * as d3 from "d3";

import type { Point } from "../models/Point";
import type { ImageSet } from "../models/ImageSet";
import type { ProcessedImage } from "../models/ProcessedImage";
import type { UploadHistoryItem } from "../models/UploadHistoryItem";

import { Header } from "../components/Header";
import { NavigationControls } from "../components/NavigationControls";
import { ImageVersionControls } from "../components/ImageVersionControls";
import { HistoryPanel } from "../components/HistoryPanel";
import { MainViewStyles } from "./MainView.style";
import { SVGViewer } from "../components/SVGViewer";
import { ApiService } from "../services/ApiService";
import { ExportService } from "../services/ExportService";
import type { AnnotationsData } from "../models/AnnotationsData";

interface MainState {
  currentImageIndex: number;
  images: ProcessedImage[];
  needsScaling: boolean;
  currentImageURL: string | null;
  loading: boolean;
  imageWidth: number;
  imageHeight: number;
  scatterData: Point[];
  downloadData: Point[];
  dataError: Error | null;
  dataLoading: boolean;
  imageFilename: string | null;
  dataFetched: boolean;
  selectedImageVersion: "original" | "inverted" | "color_contrasted";
  selectedPoint: Point | null;
  originalScatterData: Point[];
  uploadHistory: UploadHistoryItem[];
  imageSet: ImageSet;
  lizardCount: number;
  zoomTransform: d3.ZoomTransform;
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
interface MainProps {}

export class MainView extends Component<MainProps, MainState> {
  readonly svgRef = createRef<SVGSVGElement>();
  readonly zoomRef = createRef<d3.ZoomBehavior<SVGSVGElement, unknown>>();

  state: MainState = {
    currentImageIndex: 0,
    images: [],
    needsScaling: true,
    currentImageURL: null,
    loading: false,
    imageWidth: 0,
    imageHeight: 0,
    scatterData: [],
    downloadData: [],
    dataError: null,
    dataLoading: false,
    imageFilename: null,
    dataFetched: false,
    selectedImageVersion: "original",
    selectedPoint: null,
    originalScatterData: [],
    uploadHistory: [],
    imageSet: {
      original: "",
      inverted: "",
      color_contrasted: "",
    },
    lizardCount: 0,
    zoomTransform: d3.zoomIdentity,
  };

  componentDidMount(): void {
    this.fetchUploadedFiles();
    this.setupInterval();
  }

  componentWillUnmount(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
  }

  private intervalId: NodeJS.Timeout | null = null;

  // Effect to count unique images in upload folder
  componentDidUpdate(_prevProps: MainProps, prevState: MainState): void {
    if (prevState.images !== this.state.images) {
      this.countUniqueImages();
    }

    if (
      prevState.currentImageURL !== this.state.currentImageURL &&
      this.state.currentImageURL
    ) {
      this.loadImage();
    }

    if (
      this.state.currentImageURL &&
      this.state.imageWidth &&
      this.state.imageHeight &&
      this.state.originalScatterData.length > 0 &&
      (prevState.currentImageURL !== this.state.currentImageURL ||
        prevState.imageWidth !== this.state.imageWidth ||
        prevState.imageHeight !== this.state.imageHeight ||
        prevState.originalScatterData !== this.state.originalScatterData ||
        prevState.needsScaling !== this.state.needsScaling)
    ) {
      this.renderSVG();
    }

    if (prevState.selectedPoint !== this.state.selectedPoint) {
      this.updatePointSelection();
    }
  }
  private readonly countUniqueImages = (): void => {
    this.setState((prevState) => {
      const uniqueImages = new Set(prevState.images.map((img) => img.name));
      return { lizardCount: uniqueImages.size };
    });
  };

  private readonly setupInterval = (): void => {
    this.intervalId = setInterval(this.fetchUploadedFiles, 30000);
  };
  private readonly handleUpload = async (
    e: React.ChangeEvent<HTMLInputElement>
  ): Promise<void> => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    this.setState({ loading: true, dataLoading: true, dataError: null });

    try {
      const results = await ApiService.uploadMultipleImages(Array.from(files));

      const processedImages = await Promise.all(
        results.map(async (result) => {
          const imageSets = await ApiService.fetchImageSet(result.name);
          const coords = result.coords.map((coord, index: number) => ({
            ...coord,
            id: index + 1,
          }));

          return {
            name: result.name,
            coords: coords,
            originalCoords: JSON.parse(JSON.stringify(coords)), // Deep copy
            imageSets,
            timestamp: new Date().toLocaleString(), // Add timestamp for history
          };
        })
      );
      if (processedImages.length > 0) {
        const firstImage = processedImages[0];

        this.setState((prevState) => {
          // Update images with new uploads
          const updatedImages = [...prevState.images, ...processedImages];

          // Update upload history
          const newHistory = [...prevState.uploadHistory];
          processedImages.forEach((img) => {
            newHistory.push({
              name: img.name,
              timestamp: img.timestamp,
              index: updatedImages.findIndex(
                (i) => i.name === img.name && i.timestamp === img.timestamp
              ),
            });
          });

          // Set current image to the first of the new uploads
          const newImageIndex = prevState.images.length; // Index of the first new image

          return {
            images: updatedImages,
            uploadHistory: newHistory,
            currentImageIndex: newImageIndex,
            imageFilename: firstImage.name,
            originalScatterData: firstImage.originalCoords,
            scatterData: firstImage.coords,
            imageSet: firstImage.imageSets,
            currentImageURL: firstImage.imageSets.original,
            needsScaling: true,
            dataFetched: true,
            selectedPoint: null,
          };
        });
      }
    } catch (err) {
      console.error("Upload error:", err);
      this.setState({
        dataError: err instanceof Error ? err : new Error("Upload failed"),
      });
    } finally {
      this.setState({ loading: false });
      // dataLoading will be set to false by the image onload handler
    }
  };
  // This loads the image when the currentImageURL changes
  private readonly loadImage = (): void => {
    if (this.state.currentImageURL) {
      console.log("Loading image from URL:", this.state.currentImageURL);

      const img = new Image();

      img.onload = () => {
        console.log(
          "Image loaded successfully, dimensions:",
          img.width,
          "x",
          img.height
        );
        this.setState({
          imageWidth: img.width,
          imageHeight: img.height,
          dataLoading: false,
          needsScaling: true, // Reset scaling flag when new image is loaded, forcing recalculation
        });
      };

      img.onerror = (e) => {
        console.error("Failed to load image:", e);
        this.setState({
          dataError: new Error(
            "Failed to load image. Please try again with a different file."
          ),
          dataLoading: false,
        });
      };

      // Set src after defining handlers
      img.src = this.state.currentImageURL;
    }
  };

  // Renders SVG only when necessary and preserves zoom state
  private readonly renderSVG = (): void => {
    if (
      this.state.currentImageURL &&
      this.state.imageWidth &&
      this.state.imageHeight &&
      this.state.originalScatterData.length > 0
    ) {
      console.log(
        "Rendering SVG with image dimensions:",
        this.state.imageWidth,
        "x",
        this.state.imageHeight
      );

      const svg = d3.select<SVGSVGElement, unknown>(this.svgRef.current!);
      svg.selectAll("*").remove(); // Clear SVG first to prevent duplication

      // Calculate dimensions maintaining aspect ratio
      const windowHeight = window.innerHeight - window.innerHeight * 0.2;
      const width =
        windowHeight * (this.state.imageWidth / this.state.imageHeight);
      const height = windowHeight;

      svg.attr("width", width).attr("height", height);

      // Calculate scaling factors
      const xScale = width / this.state.imageWidth;
      const yScale = height / this.state.imageHeight;

      console.log("Scaling factors:", xScale, yScale);

      // Define scaling functions
      if (this.state.needsScaling) {
        console.log("Scaling scatter data to match SVG dimensions");

        // Only calculate scale once per image load
        const scaleX = d3
          .scaleLinear()
          .domain([0, this.state.imageWidth])
          .range([0, width]);

        const scaleY = d3
          .scaleLinear()
          .domain([0, this.state.imageHeight])
          .range([0, height]);

        // Store scales in refs (simulated with state for this conversion)
        this.setState((prevState) => ({
          ...prevState,
          needsScaling: false,
        })); // Always scale from original coordinates
        this.setState((prevState) => {
          const scaledData = prevState.originalScatterData.map((point) => {
            // Check if the point already has x and y values before scaling
            if (typeof point.x === "number" && typeof point.y === "number") {
              return {
                ...point,
                x: scaleX(point.x),
                y: scaleY(point.y),
              };
            } else {
              console.error("Invalid point data:", point);
              // Return a default value to avoid crashing
              return { ...point, x: 0, y: 0 };
            }
          });

          if (scaledData.length > 0) {
            console.log(
              "First original point:",
              prevState.originalScatterData[0]
            );
            console.log("First scaled data point:", scaledData[0]);
          }

          return { scatterData: scaledData, needsScaling: false };
        });
      }

      // Create container for zoom behavior
      const zoomContainer = svg.append("g").attr("class", "zoom-container");

      // Add image to the zoom container
      zoomContainer
        .append("image")
        .attr("class", "background-img")
        .attr("href", this.state.currentImageURL)
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
        .data(this.state.scatterData)
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
            this.state.selectedPoint && d.id === this.state.selectedPoint.id
              ? "yellow"
              : "red"
          )
          .attr("stroke", "black")
          .attr(
            "stroke-width",
            this.state.selectedPoint && d.id === this.state.selectedPoint.id
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

      // Add drag behavior
      pointGroups.call(
        d3
          .drag<SVGGElement, Point>()
          .on("start", this.dragstarted)
          .on("drag", this.dragged)
          .on("end", this.dragended)
      );

      // Add zoom behavior, preserving the current zoom state
      const zoom = d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.5, 5]) // Define the zoom scale extent (min, max)
        .on("zoom", (event) => {
          zoomContainer.attr("transform", event.transform.toString());
          this.setState({ zoomTransform: event.transform });
        })
        .filter((event: Event) => {
          const mouseEvent = event as MouseEvent;
          // Disable double-click zooming
          if (event.type === "dblclick") {
            event.preventDefault();
            return false;
          }
          return !mouseEvent.button && event.type !== "dblclick";
        });

      // Store the zoom reference for external control
      this.zoomRef.current = zoom;

      // Apply zoom behavior to the SVG
      svg.call(zoom);

      // Always apply the stored transform to preserve zoom state
      if (
        this.state.zoomTransform &&
        this.state.zoomTransform !== d3.zoomIdentity
      ) {
        svg.call(zoom.transform, this.state.zoomTransform);
      }
    }
  };

  private readonly dragstarted = (
    event: d3.D3DragEvent<SVGGElement, Point, Point>,
    d: Point
  ): void => {
    // Prevent event from bubbling up to zoom behavior
    event.sourceEvent.stopPropagation();

    d3.select(event.sourceEvent.target.parentNode)
      .raise()
      .attr("stroke", "black");
    this.setState({ selectedPoint: d }); // Update selected point when starting to drag    // Update visual appearance of all points
    const svg = d3.select(this.svgRef.current);
    const scatterPlotGroup = svg.select(".scatter-points");
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

    // Calculate actual coordinates accounting for zoom
    const x = (point[0] - transform.x) / transform.k;
    const y = (point[1] - transform.y) / transform.k;

    const group = d3.select(event.sourceEvent.target.parentNode);
    group.select("circle").attr("cx", x).attr("cy", y);

    group
      .select("text")
      .attr("x", x + 5)
      .attr("y", y - 5);
    this.setState((prevState) => {
      const updatedScatterData = prevState.scatterData.map((p) =>
        p.id === d.id ? { ...p, x, y } : p
      );

      // Create mock scale functions for coordinate conversion
      const scaleX = d3
        .scaleLinear()
        .domain([0, prevState.imageWidth])
        .range([
          0,
          window.innerHeight * (prevState.imageWidth / prevState.imageHeight),
        ]);

      const scaleY = d3
        .scaleLinear()
        .domain([0, prevState.imageHeight])
        .range([0, window.innerHeight - window.innerHeight * 0.2]);

      // Update original coordinates when dragging
      const updatedOriginalCoords = prevState.originalScatterData.map((p) =>
        p.id === d.id
          ? {
              ...p,
              x: scaleX.invert(x),
              y: scaleY.invert(y),
            }
          : p
      );

      return {
        scatterData: updatedScatterData,
        originalScatterData: updatedOriginalCoords,
        selectedPoint: { ...d, x, y },
      };
    });
  };

  private readonly dragended = (
    event: d3.D3DragEvent<SVGGElement, Point, Point>
  ): void => {
    // Prevent event from bubbling up to zoom behavior
    event.sourceEvent.stopPropagation();

    d3.select(event.sourceEvent.target.parentNode).attr("stroke", "black");
    // Keep the point selected after drag ends
  };
  // Separate method for updating point selection styles without re-rendering the entire SVG
  private readonly updatePointSelection = (): void => {
    if (this.svgRef.current && this.state.scatterData.length > 0) {
      const svg = d3.select(this.svgRef.current);

      // Update the visual appearance of circles based on selection
      svg
        .selectAll<SVGCircleElement, Point>("circle")
        .attr("fill", (d: Point) => {
          // Match circle by data-id attribute
          const id = d.id ?? d3.select(svg.node()).attr("data-id");
          return this.state.selectedPoint && id == this.state.selectedPoint.id
            ? "yellow"
            : "red";
        })
        .attr("stroke-width", (d: Point) => {
          const id = d.id ?? d3.select(svg.node()).attr("data-id");
          return this.state.selectedPoint && id == this.state.selectedPoint.id
            ? 2
            : 1;
        });
    }
  };

  // Handle point selection from the table without affecting zoom
  private readonly handlePointSelect = (point: Point): void => {
    this.setState({ selectedPoint: point });
  };

  private readonly handleScatterData = async (): Promise<void> => {
    try {
      this.setState({ loading: true });
      
      const result = await ExportService.exportAllData(
        this.state.images,
        this.state.currentImageIndex,
        this.state.scatterData,
        this.state.originalScatterData,
        this.createImageWithPointsBlob
      );

      if (result.failedFiles > 0) {
        alert(`Some files failed to download: ${result.failedFiles} of ${result.totalFiles}`);
      } else {
        alert(`Successfully downloaded ${result.successfulFiles} TPS file(s) and annotated image(s)`);
      }
    } catch (error) {
      alert('An error occurred during download');
      console.error('Download error:', error);
    } finally {
      this.setState({ loading: false });
    }
  };

  // Helper method to create image with points overlay as blob
  private createImageWithPointsBlob = async (imageIndex: number, imageName: string): Promise<Blob> => {
    return new Promise((resolve, reject) => {
      try {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d')!;

        // Determine which image and coordinates to use
        const imageData = imageIndex === this.state.currentImageIndex ? {
          imageUrl: this.state.currentImageURL!,
          coords: this.state.originalScatterData, // Use original coordinates instead of scaled ones
          width: this.state.imageWidth,
          height: this.state.imageHeight
        } : {
          imageUrl: this.state.images[imageIndex].imageSets.original,
          coords: this.state.images[imageIndex].originalCoords || this.state.images[imageIndex].coords,
          width: 0, // Will be set when image loads
          height: 0 // Will be set when image loads
        };

        const img = new Image();

        img.onload = () => {
          // Set canvas dimensions to match original image
          canvas.width = imageIndex === this.state.currentImageIndex ? imageData.width : img.width;
          canvas.height = imageIndex === this.state.currentImageIndex ? imageData.height : img.height;

          // Draw the background image
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

          // Draw each point
          ctx.fillStyle = 'red';
          ctx.strokeStyle = 'black';
          ctx.lineWidth = 1;

          imageData.coords.forEach((point) => {
            // Use original coordinates directly since we're working with original image dimensions
            const x = point.x;
            const y = point.y;

            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();

            // Reset styles for next point
            ctx.fillStyle = 'red';
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 1;
          });

          // Convert canvas to blob
          canvas.toBlob((blob) => {
            if (blob) {
              resolve(blob);
            } else {
              reject(new Error('Failed to create blob from canvas'));
            }
          }, 'image/png');
        };

        img.onerror = () => {
          reject(new Error(`Failed to load image: ${imageName}`));
        };

        // Set image source
        img.src = imageData.imageUrl;

      } catch (error) {
        reject(error);
      }
    });
  }

  // Format for displaying coordinates in the table
  private readonly formatCoord = (value: number): string => {
    return value ? (Math.round(value * 100) / 100).toString() : "N/A";
  };

  // Function to download TPS file directly to user's downloads
  private readonly downloadTpsFile = async (payload: {
    coords: { x: number; y: number }[];
    name: string;
  }): Promise<void> => {
    try {
      // Create TPS file content
      let tpsContent = `LM=${payload.coords.length}\n`;

      payload.coords.forEach((point) => {
        tpsContent += `${point.x} ${point.y}\n`;
      });

      tpsContent += `IMAGE=${payload.name.split(".")[0]}`;

      // Create a blob from the TPS content
      const blob = new Blob([tpsContent], { type: "text/plain" });

      // Create a download link for the blob
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      const filename = payload.name.split(".")[0] + ".tps";
      link.download = filename;

      // Trigger the download
      document.body.appendChild(link);
      link.click();

      // Clean up
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      console.log("TPS file downloaded successfully:", filename);
    } catch (error) {
      console.error("Error downloading TPS file:", error);
      throw error;
    }
  };
  // Function to download the annotated image from the backend
  private readonly downloadAnnotatedImage = async (
    imageUrl: string,
    imageName: string
  ): Promise<void> => {
    try {
      const blob = await ApiService.downloadAnnotatedImage(imageUrl);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `annotated_${imageName}`;

      // Trigger the download
      document.body.appendChild(link);
      link.click();

      // Clean up
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      console.log("Annotated image downloaded successfully:", imageName);
    } catch (error) {
      console.error("Error downloading annotated image:", error);
      throw error;
    }
  };

  // Function to download the current image with points overlay as an alternative
  private readonly downloadCurrentImageWithPoints = async (
    imageIndex: number,
    imageName: string
  ): Promise<void> => {
    try {
      // If this is the current image, use the SVG element
      if (imageIndex === this.state.currentImageIndex && this.svgRef.current) {
        // Get the current SVG element
        const svg = this.svgRef.current;

        // Create a canvas to draw the image and points
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d")!;

        // Set canvas dimensions to match the SVG
        canvas.width = svg.width.baseVal.value;
        canvas.height = svg.height.baseVal.value;

        // Get the background image
        const img = new Image();
        img.onload = () => {
          // Draw the background image
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

          // Draw each point from scatterData
          ctx.fillStyle = "red";
          ctx.strokeStyle = "black";

          this.state.scatterData.forEach((point) => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();

            // Add point ID labels
            ctx.fillStyle = "white";
            ctx.font = "10px Arial";
            ctx.fillText(point.id.toString(), point.x + 5, point.y - 5);
            ctx.fillStyle = "red"; // Reset fill color for next point
          });

          // Convert canvas to image and trigger download
          const imageUrl = canvas.toDataURL("image/png");
          const link = document.createElement("a");
          link.href = imageUrl;
          link.download = `points_overlay_${imageName}`;

          // Trigger download
          document.body.appendChild(link);
          link.click();

          // Clean up
          document.body.removeChild(link);

          console.log(
            "Image with points overlay downloaded successfully:",
            imageName
          );
        };

        // Set the image source - current displayed image
        img.src = this.state.currentImageURL!;
      } else {
        // For non-current images, create a new canvas and use the stored image data
        const img = new Image();

        // Create a promise so we can wait for the image to load
        await new Promise<void>((resolve, reject) => {
          img.onload = () => {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d")!;

            // Set canvas dimensions to match the image
            canvas.width = img.width;
            canvas.height = img.height;

            // Draw the background image
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            // Draw each point
            ctx.fillStyle = "red";
            ctx.strokeStyle = "black";

            this.state.images[imageIndex].coords.forEach((point, idx) => {
              ctx.beginPath();
              ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
              ctx.fill();
              ctx.stroke();

              // Add point ID labels
              ctx.fillStyle = "white";
              ctx.font = "10px Arial";
              ctx.fillText((idx + 1).toString(), point.x + 5, point.y - 5);
              ctx.fillStyle = "red"; // Reset fill color for next point
            });

            // Convert canvas to image and trigger download
            const imageUrl = canvas.toDataURL("image/png");
            const link = document.createElement("a");
            link.href = imageUrl;
            link.download = `points_overlay_${imageName}`;

            // Trigger download
            document.body.appendChild(link);
            link.click();

            // Clean up
            document.body.removeChild(link);

            console.log(
              "Image with points overlay downloaded successfully:",
              imageName
            );
            resolve();
          };

          img.onerror = () => {
            console.error("Failed to load image for overlay:", imageName);
            reject(new Error(`Failed to load image for overlay: ${imageName}`));
          };
        });

        // Set the image source
        img.src = this.state.images[imageIndex].imageSets.original!;
      }
    } catch (error) {
      console.error(
        "Error creating and downloading points overlay image:",
        error
      );
      throw error;
    }
  };
  // Handle changing to a different image in the set
  private readonly changeCurrentImage = (index: number): void => {
    this.setState((prevState) => {
      if (index < 0 || index >= prevState.images.length) return prevState; // Save current data to the current image before switching
      const updatedImages = [...prevState.images];
      if (
        prevState.currentImageIndex !== index &&
        prevState.images.length > 0
      ) {
        // Save both current scatter data (display) and original coordinates
        updatedImages[prevState.currentImageIndex].coords =
          prevState.scatterData;
        updatedImages[prevState.currentImageIndex].originalCoords =
          prevState.originalScatterData;
      }

      // Set new current image
      const newImage = updatedImages[index];
      return {
        ...prevState,
        images: updatedImages,
        currentImageIndex: index,
        imageFilename: newImage.name,
        imageSet: newImage.imageSets,
        currentImageURL: newImage.imageSets.original,
        needsScaling: true,
        selectedPoint: null,
        scatterData: newImage.coords,
        originalScatterData: newImage.originalCoords || newImage.coords,
      };
    });
  };
  private readonly fetchUploadedFiles = async (): Promise<void> => {
    try {
      const files = await ApiService.fetchUploadedFiles();
      console.log("Files in upload folder:", files);

      // Create history entries for files that aren't in current upload history
      const currentFileNames = new Set(
        this.state.uploadHistory.map((item) => item.name)
      );
      const newHistory = [...this.state.uploadHistory];

      files.forEach((file: string) => {
        if (!currentFileNames.has(file)) {
          newHistory.push({
            name: file,
            timestamp: "From uploads folder",
            index: -1, // Will be set when loaded
          });
        }
      });

      if (newHistory.length !== this.state.uploadHistory.length) {
        this.setState({ uploadHistory: newHistory });
      }

      // Update lizard count to include ALL files in the upload folder
      this.setState({ lizardCount: files.length });
    } catch (error) {
      console.log("Error fetching upload directory files:", error);
      // Even if fetch fails, count unique images in the current session
      this.countUniqueImages();
    }
  };
  private readonly loadImageFromUploads = async (
    filename: string
  ): Promise<void> => {
    try {
      this.setState({ loading: true, dataLoading: true });

      // Immediately set dataFetched to true to show the UI elements
      this.setState({ dataFetched: true });

      // Check if this file is already loaded in our images array
      const existingIndex = this.state.images.findIndex(
        (img) => img.name === filename
      );
      if (existingIndex >= 0) {
        this.changeCurrentImage(existingIndex);
        this.setState({ loading: false, dataLoading: false });
        return;
      }

      // If not loaded, process it
      const result = await ApiService.processExistingImage(filename);
      const imageSets = await ApiService.fetchImageSet(filename);

      const coords = result.coords.map((coord: Point, index: number) => ({
        ...coord,
        id: index + 1,
      }));

      const newImage: ProcessedImage = {
        name: filename,
        coords: coords,
        originalCoords: JSON.parse(JSON.stringify(coords)),
        imageSets,
        timestamp: "From uploads folder",
      };

      this.setState((prevState) => {
        // Add to images array
        const updatedImages = [...prevState.images, newImage];

        // Update history entry with correct index
        const newHistoryItem: UploadHistoryItem = {
          name: filename,
          timestamp: "From uploads folder",
          index: updatedImages.length - 1,
        };

        const updatedHistory = prevState.uploadHistory.map((item) =>
          item.name === filename ? newHistoryItem : item
        );

        return {
          images: updatedImages,
          uploadHistory: updatedHistory,
          currentImageIndex: updatedImages.length - 1,
          imageFilename: filename,
          scatterData: coords,
          originalScatterData: JSON.parse(JSON.stringify(coords)),
          imageSet: imageSets,
          currentImageURL: imageSets.original,
          needsScaling: true,
          selectedPoint: null,
        };
      });
    } catch (error) {
      console.error("Error loading image from uploads:", error);
      this.setState({
        dataError: error instanceof Error ? error : new Error("Unknown error"),
      });
    } finally {
      this.setState({ loading: false });
    }
  };

  private readonly handleSaveAnnotations = async (): Promise<void> => {
    if (
      !this.state.dataFetched ||
      this.state.loading ||
      !this.state.imageFilename
    )
      return;

    try {
      this.setState({ loading: true });

      // Convert scaled coordinates back to original image coordinates
      const scaleX = d3
        .scaleLinear()
        .domain([
          0,
          window.innerHeight * (this.state.imageWidth / this.state.imageHeight),
        ])
        .range([0, this.state.imageWidth]);

      const scaleY = d3
        .scaleLinear()
        .domain([0, window.innerHeight - window.innerHeight * 0.2])
        .range([0, this.state.imageHeight]);

      const originalCoords: Point[] = this.state.scatterData.map((point) => ({
        x: scaleX.invert(point.x),
        y: scaleY.invert(point.y),
        id: point.id,
      }));

      const payload: AnnotationsData = {
        coords: originalCoords,
        name: this.state.imageFilename,
      };

      // Send data to backend using ApiService
      const result = await ApiService.saveAnnotations(payload);

      // Update originalScatterData with the saved coordinates
      const updatedOriginalCoords = originalCoords.map((coord, index) => ({
        ...coord,
        id: index + 1,
      }));
      this.setState((prevState) => {
        // Update the current image's data in the images array
        const updatedImages = [...prevState.images];
        updatedImages[prevState.currentImageIndex].originalCoords =
          updatedOriginalCoords;
        updatedImages[prevState.currentImageIndex].coords =
          prevState.scatterData;

        return {
          images: updatedImages,
          originalScatterData: updatedOriginalCoords,
        };
      });

      // Show success message
      alert("Annotations saved successfully!");
      console.log("Save result:", result);
    } catch (error) {
      console.error("Error saving annotations:", error);
      alert(
        `Error saving annotations: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    } finally {
      this.setState({ loading: false });
    }
  };

  private readonly handleScatterDataUpdate = (
    scatterData: Point[],
    originalScatterData: Point[]
  ): void => {
    this.setState({ scatterData, originalScatterData });
  };

  private readonly handleScalingComplete = (): void => {
    this.setState({ needsScaling: false });
  };

  private readonly handleZoomChange = (transform: d3.ZoomTransform): void => {
    this.setState({ zoomTransform: transform });
  };

  render() {
    return (
      <div style={MainViewStyles.container}>
        {" "}
        <Header
          lizardCount={this.state.lizardCount}
          loading={this.state.loading}
          dataFetched={this.state.dataFetched}
          dataError={this.state.dataError}
          onUpload={this.handleUpload}
          onExportAll={this.handleScatterData}
        />
        <div style={MainViewStyles.mainContentArea}>
          {" "}
          <HistoryPanel
            uploadHistory={this.state.uploadHistory}
            currentImageIndex={this.state.currentImageIndex}
            onSelectImage={this.changeCurrentImage}
            onLoadFromUploads={this.loadImageFromUploads}
          />
          <div style={MainViewStyles.svgContainer}>
            {" "}
            <NavigationControls
              currentImageIndex={this.state.currentImageIndex}
              totalImages={this.state.images.length}
              loading={this.state.loading}
              onPrevious={() =>
                this.changeCurrentImage(this.state.currentImageIndex - 1)
              }
              onNext={() =>
                this.changeCurrentImage(this.state.currentImageIndex + 1)
              }
            />
            <ImageVersionControls
              dataFetched={this.state.dataFetched}
              imageSet={this.state.imageSet}
              currentImageURL={this.state.currentImageURL}
              loading={this.state.loading}
              dataLoading={this.state.dataLoading}
              onVersionChange={(imageURL: string) => {
                this.setState({
                  needsScaling: true,
                  currentImageURL: imageURL,
                });
              }}
            />{" "}
            <SVGViewer
              dataFetched={this.state.dataFetched}
              loading={this.state.loading}
              dataLoading={this.state.dataLoading}
              dataError={this.state.dataError}
              uploadHistory={this.state.uploadHistory}
              scatterData={this.state.scatterData}
              originalScatterData={this.state.originalScatterData}
              selectedPoint={this.state.selectedPoint}
              needsScaling={this.state.needsScaling}
              currentImageURL={this.state.currentImageURL}
              imageWidth={this.state.imageWidth}
              imageHeight={this.state.imageHeight}
              zoomTransform={this.state.zoomTransform}
              onPointSelect={this.handlePointSelect}
              onScatterDataUpdate={this.handleScatterDataUpdate}
              onScalingComplete={this.handleScalingComplete}
              onZoomChange={this.handleZoomChange}
            />
          </div>
        </div>
      </div>
    );
  }
}

export default MainView;
