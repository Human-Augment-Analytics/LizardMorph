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
import { SessionInfo } from "../components/SessionInfo";
import { MainViewStyles } from "./MainView.style";
import { SVGViewer } from "../components/SVGViewer";
import { ApiService } from "../services/ApiService";
import { ExportService } from "../services/ExportService";

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
  isEditMode: boolean;
  sessionReady: boolean;
  onPointSelect: (point: Point | null) => void;
  onScatterDataUpdate: (
    scatterData: Point[],
    originalScatterData: Point[]
  ) => void;
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
    isEditMode: false,
    sessionReady: false,
    onPointSelect: () => {},
    onScatterDataUpdate: () => {},
  };
  componentDidMount(): void {
    this.initializeApp();
  }

  componentWillUnmount(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
    // Clean up history on component unmount
    this.clearHistory();
    // Remove beforeunload event listener
    window.removeEventListener("beforeunload", this.handleBeforeUnload);
  }
  private async initializeApp(): Promise<void> {
    try {
      console.log("Initializing LizardMorph app...");

      // Initialize session management (will reuse existing session if available)
      await ApiService.initialize();

      // Mark session as ready
      this.setState({ sessionReady: true });
      console.log("Session initialized successfully");

      // Now proceed with normal initialization
      this.fetchUploadedFiles();
      this.setupBeforeUnloadHandler();
    } catch (error) {
      console.error("Failed to initialize app:", error);
      // Show error to user or handle gracefully
      this.setState({
        dataError:
          error instanceof Error
            ? error
            : new Error("Failed to initialize session"),
        sessionReady: false,
      });
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
        this.state.needsScaling)
    ) {
      this.renderSVG();
    }
  }
  private readonly countUniqueImages = (): void => {
    this.setState((prevState) => {
      const uniqueImages = new Set(prevState.images.map((img) => img.name));
      return { lizardCount: uniqueImages.size };
    });
  };

  private readonly handleUpload = async (
    e: React.ChangeEvent<HTMLInputElement>
  ): Promise<void> => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    this.setState({ loading: true, dataLoading: true, dataError: null });
    try {
      Promise.all(
        Array.from(files).map(async (file) => {
          // Upload the file (replace with your actual upload logic)
          const [result] = await ApiService.uploadMultipleImages([file]);
          // Process the uploaded image immediately
          const imageSets = await ApiService.fetchImageSet(result.name);
          const coords = result.coords.map((coord, index: number) => ({
            ...coord,
            id: index + 1,
          }));

          const processedImage = {
            name: result.name,
            coords: coords,
            originalCoords: JSON.parse(JSON.stringify(coords)),
            imageSets,
            timestamp: new Date().toLocaleString(),
          };

          // Update state for each processed image as it finishes
          this.setState((prevState) => {
            const updatedImages = [...prevState.images, processedImage];
            const newHistory = [
              ...prevState.uploadHistory,
              {
                name: processedImage.name,
                timestamp: processedImage.timestamp,
                index: updatedImages.length - 1,
              },
            ];
            return {
              ...prevState,
              images: updatedImages,
              uploadHistory: newHistory,
              currentImageIndex:
                prevState.images.length === 0 ? 0 : prevState.currentImageIndex,
              imageFilename:
                prevState.images.length === 0
                  ? processedImage.name
                  : prevState.imageFilename,
              originalScatterData:
                prevState.images.length === 0
                  ? processedImage.originalCoords
                  : prevState.originalScatterData,
              scatterData:
                prevState.images.length === 0
                  ? processedImage.coords
                  : prevState.scatterData,
              imageSet:
                prevState.images.length === 0
                  ? processedImage.imageSets
                  : prevState.imageSet,
              currentImageURL:
                prevState.images.length === 0
                  ? processedImage.imageSets.original
                  : prevState.currentImageURL,
              needsScaling:
                prevState.images.length === 0 ? true : prevState.needsScaling,
              dataFetched:
                prevState.images.length === 0 ? true : prevState.dataFetched,
              selectedPoint:
                prevState.images.length === 0 ? null : prevState.selectedPoint,
            };
          });
        })
      ).then(() => {
        this.setState({
          loading: false,
          dataLoading: false,
        });
      });
    } catch (err) {
      console.error("Upload error:", err);
      this.setState({
        dataError: err instanceof Error ? err : new Error("Upload failed"),
      });
    }
  };
  // This loads the image when the currentImageURL changes
  private readonly loadImage = (): void => {
    if (this.state.currentImageURL) {
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

  // Handle point selection from the table without affecting zoom
  private readonly handlePointSelect = (point: Point | null): void => {
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
        alert(
          `Some files failed to download: ${result.failedFiles} of ${result.totalFiles}`
        );
      } else {
        alert(
          `Successfully downloaded ${result.successfulFiles} TPS file(s) and annotated image(s)`
        );
      }
    } catch (error) {
      alert("An error occurred during download");
      console.error("Download error:", error);
    } finally {
      this.setState({ loading: false });
    }
  };

  private readonly handleClearHistory = async (): Promise<void> => {
    const confirmed = window.confirm(
      "Are you sure you want to clear all history? This will delete all uploaded images, processed files, and session data. This action cannot be undone."
    );

    if (confirmed) {
      try {
        this.setState({ loading: true });
        await this.clearHistory();
        alert("History cleared successfully");
      } catch (error) {
        alert("Error clearing history");
        console.error("Clear history error:", error);
      } finally {
        this.setState({ loading: false });
      }
    }
  };

  // Helper method to create image with points overlay as blob
  private createImageWithPointsBlob = async (
    imageIndex: number,
    imageName: string
  ): Promise<Blob> => {
    return new Promise((resolve, reject) => {
      try {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d")!;

        // Determine which image and coordinates to use
        const imageData =
          imageIndex === this.state.currentImageIndex
            ? {
                imageUrl: this.state.currentImageURL!,
                coords: this.state.originalScatterData,
                width: this.state.imageWidth,
                height: this.state.imageHeight,
              }
            : {
                imageUrl: this.state.images[imageIndex].imageSets.original,
                coords:
                  this.state.images[imageIndex].originalCoords ||
                  this.state.images[imageIndex].coords,
                width: 0, // Will be set when image loads
                height: 0, // Will be set when image loads
              };

        const img = new Image();

        img.onload = () => {
          // Set canvas dimensions to match original image
          canvas.width =
            imageIndex === this.state.currentImageIndex
              ? imageData.width
              : img.width;
          canvas.height =
            imageIndex === this.state.currentImageIndex
              ? imageData.height
              : img.height;

          // Draw the background image
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

          // Draw each point
          ctx.fillStyle = "red";
          ctx.strokeStyle = "black";
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
            ctx.fillStyle = "red";
            ctx.strokeStyle = "black";
            ctx.lineWidth = 1;
          });

          // Convert canvas to blob
          canvas.toBlob((blob) => {
            if (blob) {
              resolve(blob);
            } else {
              reject(new Error("Failed to create blob from canvas"));
            }
          }, "image/png");
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
  };

  // Handle changing to a different image in the set
  private readonly changeCurrentImage = (index: number): void => {
    this.setState((prevState) => {
      if (index < 0 || index >= prevState.images.length) return prevState;
      const updatedImages = [...prevState.images];
      // Always save the current state of landmarks for the current image
      updatedImages[prevState.currentImageIndex].coords = prevState.scatterData;
      updatedImages[prevState.currentImageIndex].originalCoords = prevState.originalScatterData;

      // Load the last-saved state for the new image
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

  private readonly handleToggleEditMode = (): void => {
    this.setState((prevState) => ({ isEditMode: !prevState.isEditMode }));
  };

  private readonly handleResetZoom = (): void => {
    this.setState({ zoomTransform: d3.zoomIdentity });
  };

  // Add cleanup functionality to clear history when the app closes, including beforeunload event handler
  private readonly setupBeforeUnloadHandler = (): void => {
    window.addEventListener("beforeunload", this.handleBeforeUnload);
    window.addEventListener("unload", this.handleBeforeUnload);
  };
  private readonly handleBeforeUnload = async (): Promise<void> => {
    try {
      // Clear session history before page closes
      await this.clearHistory();
    } catch (error) {
      console.error("Failed to clear session on page close:", error);
    }
  };

  private readonly clearHistory = async (): Promise<void> => {
    try {
      await ApiService.clearHistory();
      console.log("Session history cleared");
    } catch (error) {
      console.error("Failed to clear history:", error);
    }
  };
  render() {
    return (
      <div style={MainViewStyles.container}>
        {this.state.sessionReady && <SessionInfo />}
        <Header
          lizardCount={this.state.lizardCount}
          loading={this.state.loading}
          dataFetched={this.state.dataFetched}
          dataError={this.state.dataError}
          onUpload={this.handleUpload}
          onExportAll={this.handleScatterData}
          onClearHistory={this.handleClearHistory}
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
              isEditMode={this.state.isEditMode}
              onToggleEditMode={this.handleToggleEditMode}
              onResetZoom={this.handleResetZoom}
            />
            <div style={{ overflow: "auto", height: "100%" }}>
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
                isEditMode={this.state.isEditMode}
                onToggleEditMode={this.handleToggleEditMode}
                onResetZoom={this.handleResetZoom}
              />
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default MainView;
