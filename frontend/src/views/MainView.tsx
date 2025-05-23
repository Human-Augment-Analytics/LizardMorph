import React, { Component, createRef } from 'react';
import * as d3 from 'd3';

import type { Point } from "../models/Point";
import type { ImageSet } from "../models/ImageSet";
import type { ProcessedImage } from "../models/ProcessedImage";
import type { UploadHistoryItem } from "../models/UploadHistoryItem";

import { MainViewStyles } from './MainView.style';

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
  selectedImageVersion: 'original' | 'inverted' | 'color_contrasted';
  selectedPoint: Point | null;
  originalScatterData: Point[];
  uploadHistory: UploadHistoryItem[];
  imageSet: ImageSet;
  lizardCount: number;
  zoomTransform: d3.ZoomTransform;
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
interface MainProps {}

// Configuration object for API endpoints
const config = {
  apiBaseUrl: "/api",
};

export class MainView extends Component<MainProps, MainState> {
  readonly svgRef = createRef<SVGSVGElement>();
  readonly scaleXRef = createRef<d3.ScaleLinear<number, number>>();
  readonly scaleYRef = createRef<d3.ScaleLinear<number, number>>();
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
    selectedImageVersion: 'original',
    selectedPoint: null,
    originalScatterData: [],
    uploadHistory: [],
    imageSet: {
      original: "",
      inverted: "",
      color_contrasted: ""
    },
    lizardCount: 0,
    zoomTransform: d3.zoomIdentity
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
  componentDidUpdate(prevProps: MainProps, prevState: MainState): void {
    if (prevState.images !== this.state.images) {
      this.countUniqueImages();
    }

    if (prevState.currentImageURL !== this.state.currentImageURL && this.state.currentImageURL) {
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

  private countUniqueImages = (): void => {
    const uniqueImages = new Set(this.state.images.map(img => img.name));
    this.setState({ lizardCount: uniqueImages.size });
  };

  private setupInterval = (): void => {
    this.intervalId = setInterval(this.fetchUploadedFiles, 30000);
  };

  // Function to handle zoom controls
  private handleZoom = (type: 'in' | 'out' | 'reset'): void => {
    if (!this.zoomRef.current) return;

    const svg = d3.select<SVGSVGElement, unknown>(this.svgRef.current!);
    const currentTransform = this.state.zoomTransform;

    switch (type) {
      case 'in':
        this.setState({ zoomTransform: currentTransform.scale(1.2) });
        svg.transition().duration(300).call(this.zoomRef.current.transform, currentTransform.scale(1.2));
        break;
      case 'out':
        this.setState({ zoomTransform: currentTransform.scale(0.8) });
        svg.transition().duration(300).call(this.zoomRef.current.transform, currentTransform.scale(0.8));
        break;
      case 'reset':
        this.setState({ zoomTransform: d3.zoomIdentity });
        svg.transition().duration(300).call(this.zoomRef.current.transform, d3.zoomIdentity);
        break;
      default:
        break;
    }
  };

  private handleUpload = async (e: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    this.setState({ loading: true, dataLoading: true, dataError: null });

    const formData = new FormData();
    Array.from(files).forEach(file => {
      formData.append('image', file);
    });

    try {
      const response = await fetch(`${config.apiBaseUrl}/data`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const results = await response.json();

        const processedImages = await Promise.all(results.map(async (result: any) => {
          const imageSets = await this.fetchImageSet(result.name);
          const coords = result.coords.map((coord: any, index: number) => ({
            ...coord,
            id: index + 1
          }));

          return {
            name: result.name,
            coords: coords,
            originalCoords: JSON.parse(JSON.stringify(coords)), // Deep copy
            imageSets,
            timestamp: new Date().toLocaleString() // Add timestamp for history
          };
        }));

        if (processedImages.length > 0) {
          const firstImage = processedImages[0];
          // Update images with new uploads
          const updatedImages = [...this.state.images, ...processedImages];

          // Update upload history
          const newHistory = [...this.state.uploadHistory];
          processedImages.forEach(img => {
            newHistory.push({
              name: img.name,
              timestamp: img.timestamp,
              index: updatedImages.findIndex(i => i.name === img.name && i.timestamp === img.timestamp)
            });
          });

          // Set current image to the first of the new uploads
          const newImageIndex = this.state.images.length; // Index of the first new image

          this.setState({
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
            selectedPoint: null
          });
        }
      } else {
        const errorResult = await response.json();
        throw new Error(errorResult.error || 'Failed to process images');
      }
    } catch (err) {
      console.error('Upload error:', err);
      this.setState({ dataError: err instanceof Error ? err : new Error('Upload failed') });
    } finally {
      this.setState({ loading: false });
      // dataLoading will be set to false by the image onload handler
    }
  };

  private fetchImageSet = async (filename: string): Promise<ImageSet> => {
    try {
      const response = await fetch(`${config.apiBaseUrl}/image?image_filename=${encodeURIComponent(filename)}`, {
        method: 'POST',
        headers: {
          'Access-Control-Allow-Origin': '*',
        }
      });

      const result = await response.json();
      if (result.error) {
        throw new Error(result.error);
      }

      const fileExtension = filename.split('.').pop()?.toLowerCase() || '';
      const mimeType = fileExtension === 'png' ? 'image/png' :
                      fileExtension === 'gif' ? 'image/gif' :
                      'image/jpeg';

      return {
        original: `data:${mimeType};base64,${result.image3}`,
        inverted: `data:${mimeType};base64,${result.image2}`,
        color_contrasted: `data:${mimeType};base64,${result.image1}`
      };
    } catch (err) {
      console.error('Error fetching image set:', err);
      throw err;
    }
  };

  // This loads the image when the currentImageURL changes
  private loadImage = (): void => {
    if (this.state.currentImageURL) {
      console.log("Loading image from URL:", this.state.currentImageURL);

      const img = new Image();

      img.onload = () => {
        console.log("Image loaded successfully, dimensions:", img.width, "x", img.height);
        this.setState({
          imageWidth: img.width,
          imageHeight: img.height,
          dataLoading: false,
          needsScaling: true // Reset scaling flag when new image is loaded, forcing recalculation
        });
      };

      img.onerror = (e) => {
        console.error("Failed to load image:", e);
        this.setState({
          dataError: new Error('Failed to load image. Please try again with a different file.'),
          dataLoading: false
        });
      };

      // Set src after defining handlers
      img.src = this.state.currentImageURL;
    }
  };

  // Renders SVG only when necessary and preserves zoom state
  private renderSVG = (): void => {
    if (this.state.currentImageURL && this.state.imageWidth && this.state.imageHeight && this.state.originalScatterData.length > 0) {
      console.log("Rendering SVG with image dimensions:", this.state.imageWidth, "x", this.state.imageHeight);

      const svg = d3.select<SVGSVGElement, unknown>(this.svgRef.current!);
      svg.selectAll('*').remove(); // Clear SVG first to prevent duplication

      // Calculate dimensions maintaining aspect ratio
      const windowHeight = window.innerHeight - window.innerHeight * 0.2;
      const width = windowHeight * (this.state.imageWidth / this.state.imageHeight);
      const height = windowHeight;

      svg.attr('width', width)
         .attr('height', height);

      // Calculate scaling factors
      const xScale = width / this.state.imageWidth;
      const yScale = height / this.state.imageHeight;

      console.log("Scaling factors:", xScale, yScale);

      // Define scaling functions
      if (this.state.needsScaling) {
        console.log("Scaling scatter data to match SVG dimensions");

        // Only calculate scale once per image load
        const scaleX = d3.scaleLinear()
          .domain([0, this.state.imageWidth])
          .range([0, width]);

        const scaleY = d3.scaleLinear()
          .domain([0, this.state.imageHeight])
          .range([0, height]);

        // Store scales in refs (simulated with state for this conversion)
        this.setState((prevState) => ({
          ...prevState,
          needsScaling: false
        }));

        // Always scale from original coordinates
        const scaledData = this.state.originalScatterData.map(point => {
          // Check if the point already has x and y values before scaling
          if (typeof point.x === 'number' && typeof point.y === 'number') {
            return {
              ...point,
              x: scaleX(point.x),
              y: scaleY(point.y)
            };
          } else {
            console.error("Invalid point data:", point);
            // Return a default value to avoid crashing
            return { ...point, x: 0, y: 0 };
          }
        });

        if (scaledData.length > 0) {
          console.log("First original point:", this.state.originalScatterData[0]);
          console.log("First scaled data point:", scaledData[0]);
        }

        this.setState({ scatterData: scaledData });
      }

      // Create container for zoom behavior
      const zoomContainer = svg.append('g')
        .attr('class', 'zoom-container');

      // Add image to the zoom container
      zoomContainer.append('image')
        .attr('class', 'background-img')
        .attr('href', this.state.currentImageURL)
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', width)
        .attr('height', height)
        .attr('preserveAspectRatio', 'xMidYMid slice');

      // Add scatter points to the zoom container
      const scatterPlotGroup = zoomContainer.append('g')
        .attr('class', 'scatter-points');

      const pointGroups = scatterPlotGroup
        .selectAll('g')
        .data(this.state.scatterData)
        .enter()
        .append('g');

      pointGroups.each((d, i, nodes) => {
        const g = d3.select(nodes[i]);

        // Add the point
        g.append('circle')
          .attr('cx', d.x)
          .attr('cy', d.y)
          .attr('r', 3)
          .attr('fill', (this.state.selectedPoint && d.id === this.state.selectedPoint.id) ? 'yellow' : 'red')
          .attr('stroke', 'black')
          .attr('stroke-width', (this.state.selectedPoint && d.id === this.state.selectedPoint.id) ? 2 : 1)
          .attr('data-id', d.id)
          .style('cursor', 'pointer');

        // Add the number label
        g.append('text')
          .attr('x', d.x + 5)
          .attr('y', d.y - 5)
          .text(d.id)
          .attr('font-size', '10px')
          .attr('fill', 'white')
          .attr('stroke', 'black')
          .attr('stroke-width', '0.5px');
      });

      // Add drag behavior
      pointGroups.call(d3.drag<SVGGElement, Point>()
        .on("start", this.dragstarted)
        .on("drag", this.dragged)
        .on("end", this.dragended));

      // Add zoom behavior, preserving the current zoom state
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.5, 5]) // Define the zoom scale extent (min, max)
        .on('zoom', (event) => {
          zoomContainer.attr('transform', event.transform.toString());
          this.setState({ zoomTransform: event.transform });
        })
        .filter((event: any) => {
          // Disable double-click zooming
          if (event.type === 'dblclick') {
            event.preventDefault();
            return false;
          }
          return !event.button && event.type !== 'dblclick';
        });

      // Store the zoom reference for external control
      this.zoomRef.current = zoom;

      // Apply zoom behavior to the SVG
      svg.call(zoom);

      // Always apply the stored transform to preserve zoom state
      if (this.state.zoomTransform && this.state.zoomTransform !== d3.zoomIdentity) {
        svg.call(zoom.transform, this.state.zoomTransform);
      }
    }
  };

  private dragstarted = (event: d3.D3DragEvent<SVGGElement, Point, Point>, d: Point): void => {
    // Prevent event from bubbling up to zoom behavior
    event.sourceEvent.stopPropagation();

    d3.select(event.sourceEvent.target.parentNode).raise().attr("stroke", "black");
    this.setState({ selectedPoint: d }); // Update selected point when starting to drag

    // Update visual appearance of all points
    const svg = d3.select(this.svgRef.current);
    const scatterPlotGroup = svg.select('.scatter-points');
    scatterPlotGroup.selectAll('circle')
      .attr('fill', (p: any) => p.id === d.id ? 'yellow' : 'red')
      .attr('stroke-width', (p: any) => p.id === d.id ? 2 : 1);
  };

  private dragged = (event: d3.D3DragEvent<SVGGElement, Point, Point>, d: Point): void => {
    const point = d3.pointer(event, this.svgRef.current);
    const transform = d3.zoomTransform(this.svgRef.current!);

    // Calculate actual coordinates accounting for zoom
    const x = (point[0] - transform.x) / transform.k;
    const y = (point[1] - transform.y) / transform.k;

    const group = d3.select(event.sourceEvent.target.parentNode);
    group.select('circle')
      .attr("cx", x)
      .attr("cy", y);

    group.select('text')
      .attr("x", x + 5)
      .attr("y", y - 5);

    const updatedScatterData = this.state.scatterData.map((p) =>
      p.id === d.id ? { ...p, x, y } : p
    );

    // Create mock scale functions for coordinate conversion
    const scaleX = d3.scaleLinear()
      .domain([0, this.state.imageWidth])
      .range([0, window.innerHeight * (this.state.imageWidth / this.state.imageHeight)]);
    
    const scaleY = d3.scaleLinear()
      .domain([0, this.state.imageHeight])
      .range([0, window.innerHeight - window.innerHeight * 0.2]);

    // Update original coordinates when dragging
    const updatedOriginalCoords = this.state.originalScatterData.map((p) =>
      p.id === d.id ? {
        ...p,
        x: scaleX.invert(x),
        y: scaleY.invert(y)
      } : p
    );

    this.setState({
      scatterData: updatedScatterData,
      originalScatterData: updatedOriginalCoords,
      selectedPoint: { ...d, x, y }
    });
  };

  private dragended = (event: d3.D3DragEvent<SVGGElement, Point, Point>, d: Point): void => {
    // Prevent event from bubbling up to zoom behavior
    event.sourceEvent.stopPropagation();

    d3.select(event.sourceEvent.target.parentNode).attr("stroke", "black");
    // Keep the point selected after drag ends
  };

  // Separate method for updating point selection styles without re-rendering the entire SVG
  private updatePointSelection = (): void => {
    if (this.svgRef.current && this.state.scatterData.length > 0) {
      const svg = d3.select(this.svgRef.current);

      // Update the visual appearance of circles based on selection
      svg.selectAll('circle')
        .attr('fill', (d: any) => {
          // Match circle by data-id attribute
          const id = d.id || d3.select(d).attr('data-id');
          return this.state.selectedPoint && id == this.state.selectedPoint.id ? 'yellow' : 'red';
        })
        .attr('stroke-width', (d: any) => {
          const id = d.id || d3.select(d).attr('data-id');
          return this.state.selectedPoint && id == this.state.selectedPoint.id ? 2 : 1;
        });
    }
  };

  // Handle point selection from the table without affecting zoom
  private handlePointSelect = (point: Point): void => {
    this.setState({ selectedPoint: point });
  };

  private handleScatterData = async (): Promise<void> => {
    // If we have multiple images, we'll download data for all of them
    const downloadPromises: Promise<string>[] = [];

    for (let i = 0; i < this.state.images.length; i++) {
      // Get the original coordinates for this image
      const originalCoords = i === this.state.currentImageIndex ?
        this.state.originalScatterData :
        this.state.images[i].originalCoords || this.state.images[i].coords;

      const payload = {
        coords: originalCoords.map(({ id, ...rest }) => rest), // Remove the id field before sending
        name: this.state.images[i].name
      };

      // Create a promise for each image's data processing
      const processPromise = (async (): Promise<string> => {
        try {
          // Send data to backend
          const response = await fetch(`${config.apiBaseUrl}/endpoint`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
          });

          if (!response.ok) {
            throw new Error(`HTTP error for ${payload.name}! Status: ${response.status}`);
          }

          const result = await response.json();
          console.log(`Downloaded data for ${payload.name}:`, result);

          // Download the TPS file to user's downloads folder
          await this.downloadTpsFile(payload);

          // Also download the annotated image if it exists
          if (result.image_urls && result.image_urls.length > 0) {
            // Convert relative URL to absolute URL using the API base
            const imageUrl = result.image_urls[0].startsWith('http')
              ? result.image_urls[0]
              : `${config.apiBaseUrl}${result.image_urls[0].startsWith('/') ? '' : '/'}${result.image_urls[0]}`;

            await this.downloadAnnotatedImage(imageUrl, payload.name);
          } else {
            // If no image URL is provided, we generate a simple visualization
            // using the current image and points data
            await this.downloadCurrentImageWithPoints(i, payload.name);
          }

          return payload.name;
        } catch (error) {
          console.error(`Error during the fetch request for ${payload.name}:`, error);
          throw error;
        }
      })();

      downloadPromises.push(processPromise);
    }

    // Wait for all downloads to complete
    try {
      this.setState({ loading: true });
      const results = await Promise.allSettled(downloadPromises);

      // Check for failures
      const failures = results.filter(r => r.status === 'rejected');

      if (failures.length > 0) {
        alert(`Some files failed to download: ${failures.length} of ${this.state.images.length}`);
      } else {
        alert(`Successfully downloaded ${this.state.images.length} TPS file(s) and annotated image(s)`);
      }
    } catch (error) {
      alert('An error occurred during download');
      console.error('Download error:', error);
    } finally {
      this.setState({ loading: false });
    }
  };

  // Format for displaying coordinates in the table
  private formatCoord = (value: number): string => {
    return value ? (Math.round(value * 100) / 100).toString() : 'N/A' as any;
  };

  // Function to download TPS file directly to user's downloads
  private downloadTpsFile = async (payload: { coords: {x: number, y: number}[], name: string }): Promise<void> => {
    try {
      // Create TPS file content
      let tpsContent = `LM=${payload.coords.length}\n`;

      payload.coords.forEach(point => {
        tpsContent += `${point.x} ${point.y}\n`;
      });

      tpsContent += `IMAGE=${payload.name.split('.')[0]}`;

      // Create a blob from the TPS content
      const blob = new Blob([tpsContent], { type: 'text/plain' });

      // Create a download link for the blob
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      const filename = payload.name.split('.')[0] + '.tps';
      link.download = filename;

      // Trigger the download
      document.body.appendChild(link);
      link.click();

      // Clean up
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      console.log('TPS file downloaded successfully:', filename);
    } catch (error) {
      console.error('Error downloading TPS file:', error);
      throw error;
    }
  };

  // Function to download the annotated image from the backend
  private downloadAnnotatedImage = async (imageUrl: string, imageName: string): Promise<void> => {
    try {
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch annotated image: ${response.status}`);
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `annotated_${imageName}`;

      // Trigger the download
      document.body.appendChild(link);
      link.click();

      // Clean up
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      console.log('Annotated image downloaded successfully:', imageName);
    } catch (error) {
      console.error('Error downloading annotated image:', error);
      throw error;
    }
  };

  // Function to download the current image with points overlay as an alternative
  private downloadCurrentImageWithPoints = async (imageIndex: number, imageName: string): Promise<void> => {
    try {
      // If this is the current image, use the SVG element
      if (imageIndex === this.state.currentImageIndex && this.svgRef.current) {
        // Get the current SVG element
        const svg = this.svgRef.current;

        // Create a canvas to draw the image and points
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d')!;

        // Set canvas dimensions to match the SVG
        canvas.width = svg.width.baseVal.value;
        canvas.height = svg.height.baseVal.value;

        // Get the background image
        const img = new Image();
        img.onload = () => {
          // Draw the background image
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

          // Draw each point from scatterData
          ctx.fillStyle = 'red';
          ctx.strokeStyle = 'black';

          this.state.scatterData.forEach(point => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();

            // Add point ID labels
            ctx.fillStyle = 'white';
            ctx.font = '10px Arial';
            ctx.fillText(point.id.toString(), point.x + 5, point.y - 5);
            ctx.fillStyle = 'red'; // Reset fill color for next point
          });

          // Convert canvas to image and trigger download
          const imageUrl = canvas.toDataURL('image/png');
          const link = document.createElement('a');
          link.href = imageUrl;
          link.download = `points_overlay_${imageName}`;

          // Trigger download
          document.body.appendChild(link);
          link.click();

          // Clean up
          document.body.removeChild(link);

          console.log('Image with points overlay downloaded successfully:', imageName);
        };

        // Set the image source - current displayed image
        img.src = this.state.currentImageURL!;
      } else {
        // For non-current images, create a new canvas and use the stored image data
        const img = new Image();

        // Create a promise so we can wait for the image to load
        await new Promise<void>((resolve, reject) => {
          img.onload = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d')!;

            // Set canvas dimensions to match the image
            canvas.width = img.width;
            canvas.height = img.height;

            // Draw the background image
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            // Draw each point
            ctx.fillStyle = 'red';
            ctx.strokeStyle = 'black';

            this.state.images[imageIndex].coords.forEach((point, idx) => {
              ctx.beginPath();
              ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
              ctx.fill();
              ctx.stroke();

              // Add point ID labels
              ctx.fillStyle = 'white';
              ctx.font = '10px Arial';
              ctx.fillText((idx + 1).toString(), point.x + 5, point.y - 5);
              ctx.fillStyle = 'red'; // Reset fill color for next point
            });

            // Convert canvas to image and trigger download
            const imageUrl = canvas.toDataURL('image/png');
            const link = document.createElement('a');
            link.href = imageUrl;
            link.download = `points_overlay_${imageName}`;

            // Trigger download
            document.body.appendChild(link);
            link.click();

            // Clean up
            document.body.removeChild(link);

            console.log('Image with points overlay downloaded successfully:', imageName);
            resolve();
          };

          img.onerror = () => {
            console.error('Failed to load image for overlay:', imageName);
            reject(new Error(`Failed to load image for overlay: ${imageName}`));
          };
        });

        // Set the image source
        img.src = this.state.images[imageIndex].imageSets.original!;
      }
    } catch (error) {
      console.error('Error creating and downloading points overlay image:', error);
      throw error;
    }
  };

  // Handle changing to a different image in the set
  private changeCurrentImage = (index: number): void => {
    if (index < 0 || index >= this.state.images.length) return;

    // Save current data to the current image before switching
    if (this.state.currentImageIndex !== index && this.state.images.length > 0) {
      const updatedImages = [...this.state.images];

      // Save both current scatter data (display) and original coordinates
      updatedImages[this.state.currentImageIndex].coords = this.state.scatterData;
      updatedImages[this.state.currentImageIndex].originalCoords = this.state.originalScatterData;

      this.setState({ images: updatedImages });
    }

    // Set new current image
    const newImage = this.state.images[index];
    this.setState({
      currentImageIndex: index,
      imageFilename: newImage.name,
      imageSet: newImage.imageSets,
      currentImageURL: newImage.imageSets.original,
      needsScaling: true,
      selectedPoint: null,
      scatterData: newImage.coords,
      originalScatterData: newImage.originalCoords || newImage.coords,
    });
  };

  private fetchUploadedFiles = async (): Promise<void> => {
    try {
      const response = await fetch(`${config.apiBaseUrl}/list_uploads`, {
        method: 'GET',
      });

      if (response.ok) {
        const files = await response.json();
        console.log('Files in upload folder:', files);

        // Create history entries for files that aren't in current upload history
        const currentFileNames = new Set(this.state.uploadHistory.map(item => item.name));
        const newHistory = [...this.state.uploadHistory];

        files.forEach((file: string) => {
          if (!currentFileNames.has(file)) {
            newHistory.push({
              name: file,
              timestamp: 'From uploads folder',
              index: -1, // Will be set when loaded
            });
          }
        });

        if (newHistory.length !== this.state.uploadHistory.length) {
          this.setState({ uploadHistory: newHistory });
        }

        // Update lizard count to include ALL files in the upload folder
        this.setState({ lizardCount: files.length });
      }
    } catch (error) {
      console.log('Error fetching upload directory files:', error);
      // Even if fetch fails, count unique images in the current session
      this.countUniqueImages();
    }
  };

  private loadImageFromUploads = async (filename: string): Promise<void> => {
    try {
      this.setState({ loading: true, dataLoading: true });

      // Immediately set dataFetched to true to show the UI elements
      this.setState({ dataFetched: true });

      // Check if this file is already loaded in our images array
      const existingIndex = this.state.images.findIndex(img => img.name === filename);
      if (existingIndex >= 0) {
        this.changeCurrentImage(existingIndex);
        this.setState({ loading: false, dataLoading: false });
        return;
      }

      // If not loaded, process it
      const response = await fetch(`${config.apiBaseUrl}/process_existing?filename=${encodeURIComponent(filename)}`, {
        method: 'POST',
      });

      if (response.ok) {
        const result = await response.json();
        const imageSets = await this.fetchImageSet(filename);

        const coords = result.coords.map((coord: any, index: number) => ({
          ...coord,
          id: index + 1,
        }));

        const newImage: ProcessedImage = {
          name: filename,
          coords: coords,
          originalCoords: JSON.parse(JSON.stringify(coords)),
          imageSets,
          timestamp: 'From uploads folder',
        };

        // Add to images array
        const updatedImages = [...this.state.images, newImage];

        // Update history entry with correct index
        const newHistoryItem: UploadHistoryItem = {
          name: filename,
          timestamp: 'From uploads folder',
          index: updatedImages.length - 1,
        };

        const updatedHistory = this.state.uploadHistory.map(item =>
          item.name === filename ? newHistoryItem : item
        );

        this.setState({
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
        });
      } else {
        throw new Error('Failed to process existing image');
      }
    } catch (error) {
      console.error('Error loading image from uploads:', error);
      this.setState({ dataError: error instanceof Error ? error : new Error('Unknown error') });
    } finally {
      this.setState({ loading: false });
    }
  };

  private handleSaveAnnotations = async (): Promise<void> => {
    if (!this.state.dataFetched || this.state.loading || !this.state.imageFilename) return;

    try {
      this.setState({ loading: true });

      // Convert scaled coordinates back to original image coordinates
      const scaleX = d3.scaleLinear()
        .domain([0, window.innerHeight * (this.state.imageWidth / this.state.imageHeight)])
        .range([0, this.state.imageWidth]);

      const scaleY = d3.scaleLinear()
        .domain([0, window.innerHeight - window.innerHeight * 0.2])
        .range([0, this.state.imageHeight]);

      const originalCoords = this.state.scatterData.map(point => ({
        x: scaleX.invert(point.x),
        y: scaleY.invert(point.y),
      }));

      const payload = {
        coords: originalCoords,
        name: this.state.imageFilename,
      };

      // Send data to backend
      const response = await fetch(`${config.apiBaseUrl}/save_annotations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const result = await response.json();

      // Update originalScatterData with the saved coordinates
      const updatedOriginalCoords = originalCoords.map((coord, index) => ({
        ...coord,
        id: index + 1,
      }));

      // Update the current image's data in the images array
      const updatedImages = [...this.state.images];
      updatedImages[this.state.currentImageIndex].originalCoords = updatedOriginalCoords;
      updatedImages[this.state.currentImageIndex].coords = this.state.scatterData;

      this.setState({
        images: updatedImages,
        originalScatterData: updatedOriginalCoords,
      });

      // Show success message
      alert('Annotations saved successfully!');
      console.log('Save result:', result);
    } catch (error) {
      console.error('Error saving annotations:', error);
      alert(`Error saving annotations: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      this.setState({ loading: false });
    }
  };

  render() {
    return (
      <div style={MainViewStyles.container}>
        <div style={MainViewStyles.header}>
          <div style={MainViewStyles.infoBox}>
            <div style={MainViewStyles.infoBoxContent}>
              <p style={MainViewStyles.infoBoxParagraph}>Made with ❤️ by the Human Augmented Analytics Group (HAAG)</p>
              <p style={MainViewStyles.infoBoxParagraph}>In Partnership with Dr. Stroud</p>
              <p style={MainViewStyles.infoBoxParagraph}>Author: Mercedes Quintana</p>
              <p style={MainViewStyles.infoBoxParagraph}>AI Engineer: Anthony Trevino</p>
              <p style={MainViewStyles.infoBoxItalic}>Georgia Institute of Technology - Spring 2025</p>
              <a 
                href="https://github.com/Human-Augment-Analytics/Lizard-CV-Web-App" 
                target="_blank" 
                rel="noopener noreferrer"
                style={MainViewStyles.infoBoxLink}
              >
                View on GitHub
              </a>
              <div style={MainViewStyles.lizardCount}>
                <strong>Number of Lizards Analyzed: {this.state.lizardCount}</strong>
              </div>
            </div>
          </div>
          
          <div style={MainViewStyles.mainContent}>
            <div style={MainViewStyles.buttonContainer}>
              <label 
                htmlFor="file-upload" 
                style={{
                  ...MainViewStyles.uploadButton,
                  ...(this.state.loading ? MainViewStyles.uploadButtonDisabled : {}),
                }}
              >
                {this.state.loading ? 'Uploading...' : 'Upload X-Ray Images'}
              </label>
              <input 
                id="file-upload" 
                type="file" 
                accept="image/*" 
                onChange={this.handleUpload}
                style={{ display: 'none' }}
                multiple
                disabled={this.state.loading}
              />
              
              <button
                onClick={this.handleScatterData}
                disabled={!this.state.dataFetched || this.state.loading}
                style={{
                  ...MainViewStyles.exportButton,
                  ...((!this.state.dataFetched || this.state.loading) ? MainViewStyles.exportButtonDisabled : {}),
                }}
              >
                Export All Data
              </button>
            </div>
            
            <div style={MainViewStyles.titleContainer}>
              <img 
                src="/android-chrome-192x192.png" 
                alt="Lizard Logo" 
                style={MainViewStyles.logo} 
              />
              <h2 style={MainViewStyles.title}>Lizard Anolis X-Ray Auto-Annotator</h2>
            </div>
            
            <div style={MainViewStyles.rightSpacer}></div>
          </div>

          {this.state.dataError && <span style={MainViewStyles.errorMessage}>Error: {this.state.dataError.message}</span>}

          {this.state.images.length > 1 && (
            <div style={MainViewStyles.navigationControls}>
              <button
                onClick={() => this.changeCurrentImage(this.state.currentImageIndex - 1)}
                disabled={this.state.currentImageIndex === 0 || this.state.loading}
                style={{
                  ...MainViewStyles.navButton,
                  ...(this.state.currentImageIndex === 0 || this.state.loading ? MainViewStyles.navButtonDisabled : {}),
                }}
              >
                Previous Image
              </button>
              
              <span>Image {this.state.currentImageIndex + 1} of {this.state.images.length}</span>
              
              <button
                onClick={() => this.changeCurrentImage(this.state.currentImageIndex + 1)}
                disabled={this.state.currentImageIndex === this.state.images.length - 1 || this.state.loading}
                style={{
                  ...MainViewStyles.navButton,
                  ...(this.state.currentImageIndex === this.state.images.length - 1 || this.state.loading ? MainViewStyles.navButtonDisabled : {}),
                }}
              >
                Next Image
              </button>
            </div>
          )}

          {this.state.dataFetched && this.state.imageSet.original && (
            <div style={MainViewStyles.imageVersionButtons}>
              <button 
                onClick={() => {
                  this.setState({ needsScaling: true, currentImageURL: this.state.imageSet.original });
                }}
                disabled={this.state.loading || this.state.dataLoading}
                style={{
                  ...MainViewStyles.versionButton,
                  ...(this.state.currentImageURL === this.state.imageSet.original ? MainViewStyles.versionButtonActive : {}),
                  ...(this.state.loading || this.state.dataLoading ? MainViewStyles.versionButtonDisabled : {}),
                }}
              >
                Original
              </button>
              <button 
                onClick={() => {
                  this.setState({ needsScaling: true, currentImageURL: this.state.imageSet.inverted });
                }}
                disabled={this.state.loading || this.state.dataLoading}
                style={{
                  ...MainViewStyles.versionButton,
                  ...(this.state.currentImageURL === this.state.imageSet.inverted ? MainViewStyles.versionButtonActive : {}),
                  ...(this.state.loading || this.state.dataLoading ? MainViewStyles.versionButtonDisabled : {}),
                }}
              >
                Inverted
              </button>
              <button 
                onClick={() => {
                  this.setState({ needsScaling: true, currentImageURL: this.state.imageSet.color_contrasted });
                }}
                disabled={this.state.loading || this.state.dataLoading}
                style={{
                  ...MainViewStyles.versionButton,
                  ...(this.state.currentImageURL === this.state.imageSet.color_contrasted ? MainViewStyles.versionButtonActive : {}),
                  ...(this.state.loading || this.state.dataLoading ? MainViewStyles.versionButtonDisabled : {}),
                }}
              >
                Color Contrasted
              </button>
            </div>
          )}
        </div>

        <div style={MainViewStyles.mainContentArea}>
          <div style={MainViewStyles.historyContainer}>
            <h3>History</h3>
            <div style={MainViewStyles.historyTableContainer}>
              <table style={MainViewStyles.historyTable}>
                <thead>
                  <tr style={MainViewStyles.historyTableHeader}>
                    <th style={MainViewStyles.historyTableHeaderCell}>Image Name</th>
                  </tr>
                </thead>
                <tbody>
                  {this.state.uploadHistory.length > 0 ? (
                    this.state.uploadHistory.map((item, idx) => (
                      <tr 
                        key={idx}
                        onClick={() => item.index >= 0 ? this.changeCurrentImage(item.index) : this.loadImageFromUploads(item.name)}
                        style={{
                          ...MainViewStyles.historyTableRow,
                          ...(item.index === this.state.currentImageIndex ? MainViewStyles.historyTableRowSelected : {}),
                        }}
                      >
                        <td style={{
                          ...MainViewStyles.historyTableCell,
                          ...(item.index === this.state.currentImageIndex ? MainViewStyles.historyTableCellSelected : {}),
                        }}>
                          {item.name}
                          <div style={{ fontSize: '0.8em', color: '#666' }}>{item.timestamp}</div>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td style={MainViewStyles.historyTableEmptyCell}>
                        No images in history
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
          
          <div style={MainViewStyles.svgContainer}>
            {!this.state.dataFetched && !this.state.loading && this.state.uploadHistory.length === 0 && (
              <div style={MainViewStyles.placeholderMessage}>
                <p>Upload one or more X-ray images to begin analysis</p>
                <p style={MainViewStyles.placeholderSubtext}>The images will appear here</p>
              </div>
            )}
            
            <svg
              ref={this.svgRef}
              style={{
                ...MainViewStyles.svg,
                ...(this.state.dataFetched ? MainViewStyles.svgWithData : {}),
              }}
            />
            
            {this.state.dataLoading && this.state.dataFetched && (
              <div style={MainViewStyles.loadingOverlay}>
                Loading image...
              </div>
            )}
            
            {this.state.dataError && !this.state.loading && (
              <div style={MainViewStyles.errorOverlay}>
                Error: {this.state.dataError.message}
              </div>
            )}
          </div>
          
          {this.state.dataFetched && (
            <div style={MainViewStyles.pointsContainer}>
              <div style={MainViewStyles.pointsHeader}>
                <h3 style={MainViewStyles.selectedPointHeader}>Landmark Points</h3>
                <button
                  onClick={this.handleSaveAnnotations}
                  disabled={this.state.loading}
                  style={{
                    ...MainViewStyles.saveButton,
                    ...(this.state.loading ? MainViewStyles.saveButtonDisabled : {}),
                  }}
                >
                  {this.state.loading ? 'Saving...' : 'Save Annotations'}
                </button>
              </div>
              
              {this.state.selectedPoint && (
                <div style={MainViewStyles.selectedPointDetails}>
                  <h4 style={MainViewStyles.selectedPointHeader}>Selected Point Details</h4>
                  <p><strong>Point {this.state.selectedPoint.id}</strong></p>
                  <p><strong>X coordinate:</strong> {this.formatCoord(this.state.selectedPoint.x)}</p>
                  <p><strong>Y coordinate:</strong> {this.formatCoord(this.state.selectedPoint.y)}</p>
                  <div style={MainViewStyles.selectedPointInfo}>
                    <p>Image: {this.state.imageFilename}</p>
                    <p>Image {this.state.currentImageIndex + 1} of {this.state.images.length}</p>
                  </div>
                </div>
              )}

              <p>Click on a row to select a point. Selected point is highlighted in yellow.</p>
              <table style={MainViewStyles.pointsTable}>
                <thead>
                  <tr style={MainViewStyles.pointsTableHeader}>
                    <th style={MainViewStyles.pointsTableHeaderCell}>Point ID</th>
                    <th style={MainViewStyles.pointsTableHeaderCell}>X</th>
                    <th style={MainViewStyles.pointsTableHeaderCell}>Y</th>
                  </tr>
                </thead>
                <tbody>
                  {this.state.scatterData.map((point) => (
                    <tr 
                      key={point.id} 
                      onClick={() => this.handlePointSelect(point)}
                      style={{
                        ...MainViewStyles.pointsTableRow,
                        ...(this.state.selectedPoint && this.state.selectedPoint.id === point.id ? MainViewStyles.pointsTableRowSelected : {}),
                      }}
                    >
                      <td style={MainViewStyles.pointsTableCell}>Point {point.id}</td>
                      <td style={MainViewStyles.pointsTableCell}>{this.formatCoord(point.x)}</td>
                      <td style={MainViewStyles.pointsTableCell}>{this.formatCoord(point.y)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    );
  }
}

export default MainView;
