import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';

// Configuration object for API endpoints
const config = {
  // Use the local Flask server instead of Azure
  apiBaseUrl: process.env.REACT_APP_API_URL || 'http://localhost:5000',
};

function App() {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [images, setImages] = useState([]);
  const [needsScaling, setNeedsScaling] = useState(true);
  const [currentImageURL, setCurrentImageURL] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imageWidth, setImageWidth] = useState(0);
  const [imageHeight, setImageHeight] = useState(0);
  const [scatterData, setScatterData] = useState([]);
  const [downloadData, setDownloadData] = useState([]);
  const [dataError, setDataError] = useState(null);
  const [dataLoading, setDataLoading] = useState(false);
  const svgRef = useRef(null);
  const [imageFilename, setImageFilename] = useState(null);
  const [dataFetched, setDataFetched] = useState(false);
  const scaleXRef = useRef(null);
  const scaleYRef = useRef(null);
  const [selectedImageVersion, setSelectedImageVersion] = useState('original');
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [originalScatterData, setOriginalScatterData] = useState([]);
  const [uploadHistory, setUploadHistory] = useState([]);

  const [imageSet, setImageSet] = useState({
    original: null,
    inverted: null,
    color_contrasted: null
  });

  const [lizardCount, setLizardCount] = useState(0);

  // Effect to count unique images in upload folder
  useEffect(() => {
    const countUniqueImages = () => {
      const uniqueImages = new Set(images.map(img => img.name));
      setLizardCount(uniqueImages.size);
    };
    countUniqueImages();
  }, [images]);


  const [zoomTransform, setZoomTransform] = useState(d3.zoomIdentity);
  const zoomRef = useRef(null);

  // Function to handle zoom controls
  const handleZoom = (type) => {
    if (!zoomRef.current) return;
    
    const svg = d3.select(svgRef.current);
    const currentTransform = zoomTransform;
    
    switch (type) {
      case 'in':
        setZoomTransform(currentTransform.scale(1.2));
        svg.transition().duration(300).call(zoomRef.current.transform, currentTransform.scale(1.2));
        break;
      case 'out':
        setZoomTransform(currentTransform.scale(0.8));
        svg.transition().duration(300).call(zoomRef.current.transform, currentTransform.scale(0.8));
        break;
      case 'reset':
        setZoomTransform(d3.zoomIdentity);
        svg.transition().duration(300).call(zoomRef.current.transform, d3.zoomIdentity);
        break;
      default:
        break;
    }
  };

  const handleUpload = async (e) => {
    const files = e.target.files;
    if (files.length === 0) return;
  
    setLoading(true);
    setDataLoading(true);
    setDataError(null);
  
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
        
        const processedImages = await Promise.all(results.map(async (result) => {
          const imageSets = await fetchImageSet(result.name);
          const coords = result.coords.map((coord, index) => ({
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
          const updatedImages = [...images, ...processedImages];
          setImages(updatedImages);
          
          // Update upload history
          const newHistory = [...uploadHistory];
          processedImages.forEach(img => {
            newHistory.push({
              name: img.name,
              timestamp: img.timestamp,
              index: updatedImages.findIndex(i => i.name === img.name && i.timestamp === img.timestamp)
            });
          });
          setUploadHistory(newHistory);
          
          // Set current image to the first of the new uploads
          const newImageIndex = images.length; // Index of the first new image
          setCurrentImageIndex(newImageIndex);
          setImageFilename(firstImage.name);
          setOriginalScatterData(firstImage.originalCoords); // Store original
          setScatterData(firstImage.coords);
          setImageSet(firstImage.imageSets);
          setCurrentImageURL(firstImage.imageSets.original);
          setNeedsScaling(true);
          setDataFetched(true);
          setSelectedPoint(null);
        }
      } else {
        const errorResult = await response.json();
        throw new Error(errorResult.error || 'Failed to process images');
      }
    } catch (err) {
      console.error('Upload error:', err);
      setDataError(err);
    } finally {
      setLoading(false);
      // dataLoading will be set to false by the image onload handler
    }
  };
  

  const fetchImageSet = async (filename) => {
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
  
      const fileExtension = filename.split('.').pop().toLowerCase();
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

  // This effect loads the image when the currentImageURL changes
  useEffect(() => {
    if (currentImageURL) {
      console.log("Loading image from URL:", currentImageURL);
      
      const img = new Image();
      
      img.onload = () => {
        console.log("Image loaded successfully, dimensions:", img.width, "x", img.height);
        setImageWidth(img.width);
        setImageHeight(img.height);
        setDataLoading(false);
        
        // Reset scaling flag when new image is loaded, forcing recalculation
        setNeedsScaling(true);
      };
      
      img.onerror = (e) => {
        console.error("Failed to load image:", e);
        setDataError(new Error('Failed to load image. Please try again with a different file.'));
        setDataLoading(false);
      };
      
      // Set src after defining handlers
      img.src = currentImageURL;
    }
  }, [currentImageURL]);

  // FIX: This renders SVG only when necessary and preserves zoom state
  useEffect(() => {
    if (currentImageURL && imageWidth && imageHeight && originalScatterData.length > 0) {
      console.log("Rendering SVG with image dimensions:", imageWidth, "x", imageHeight);
      
      const svg = d3.select(svgRef.current);
      svg.selectAll('*').remove(); // Clear SVG first to prevent duplication
      
      // Calculate dimensions maintaining aspect ratio
      const windowHeight = window.innerHeight - window.innerHeight * 0.2;
      const width = windowHeight * (imageWidth / imageHeight);
      const height = windowHeight;
      
      svg.attr('width', width)
         .attr('height', height);
      
      // Calculate scaling factors
      const xScale = width / imageWidth;
      const yScale = height / imageHeight;
      
      console.log("Scaling factors:", xScale, yScale);
      
      // Define scaling functions
      if (needsScaling) {
        console.log("Scaling scatter data to match SVG dimensions");
        
        // Only calculate scale once per image load
        scaleXRef.current = d3.scaleLinear()
          .domain([0, imageWidth])
          .range([0, width]);
        
        scaleYRef.current = d3.scaleLinear()
          .domain([0, imageHeight])
          .range([0, height]);
        
        // Always scale from original coordinates
        const scaledData = originalScatterData.map(point => {
          // Check if the point already has x and y values before scaling
          if (typeof point.x === 'number' && typeof point.y === 'number') {
            return {
              ...point,
              x: scaleXRef.current(point.x),
              y: scaleYRef.current(point.y)
            };
          } else {
            console.error("Invalid point data:", point);
            // Return a default value to avoid crashing
            return { ...point, x: 0, y: 0 };
          }
        });
        
        if (scaledData.length > 0) {
          console.log("First original point:", originalScatterData[0]);
          console.log("First scaled data point:", scaledData[0]);
        }
        
        setScatterData(scaledData);
        setNeedsScaling(false);
      }
      
      // Create container for zoom behavior
      const zoomContainer = svg.append('g')
        .attr('class', 'zoom-container');
      
      // Add image to the zoom container
      zoomContainer.append('image')
        .attr('class', 'background-img')
        .attr('href', currentImageURL)  
        .attr('x', 0)            
        .attr('y', 0)          
        .attr('width', width)    
        .attr('height', height) 
        .attr('preserveAspectRatio', 'xMidYMid slice');
  
      // Add scatter points to the zoom container
      const scatterPlotGroup = zoomContainer.append('g')
        .attr('class', 'scatter-points');
  
      scatterPlotGroup
        .selectAll('g')
        .data(scatterData)
        .enter()
        .append('g')
        .each(function(d) {
          const g = d3.select(this);
          
          // Add the point
          g.append('circle')
            .attr('cx', d.x)
            .attr('cy', d.y)
            .attr('r', 3)
            .attr('fill', d => (selectedPoint && d.id === selectedPoint.id) ? 'yellow' : 'red')
            .attr('stroke', 'black')
            .attr('stroke-width', d => (selectedPoint && d.id === selectedPoint.id) ? 2 : 1)
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
        })
        .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended));
  
      function dragstarted(event, d) {
        // Prevent event from bubbling up to zoom behavior
        event.sourceEvent.stopPropagation();
        
        d3.select(this).raise().attr("stroke", "black");
        setSelectedPoint(d); // Update selected point when starting to drag
        
        // Update visual appearance of all points
        scatterPlotGroup.selectAll('circle')
          .attr('fill', p => p.id === d.id ? 'yellow' : 'red')
          .attr('stroke-width', p => p.id === d.id ? 2 : 1);
      }
      
      function dragged(event, d) {
        const point = d3.pointer(event, svg.node());
        const transform = d3.zoomTransform(svg.node());
        
        // Calculate actual coordinates accounting for zoom
        const x = (point[0] - transform.x) / transform.k;
        const y = (point[1] - transform.y) / transform.k;
        
        d3.select(this)
          .select('circle')
          .attr("cx", x)
          .attr("cy", y);
          
        d3.select(this)
          .select('text')
          .attr("x", x + 5)
          .attr("y", y - 5);
      
        const updatedScatterData = scatterData.map((p) =>
          p.id === d.id ? { ...p, x, y } : p
        );
        
        setScatterData(updatedScatterData);
        
        // Update original coordinates when dragging
        const updatedOriginalCoords = originalScatterData.map((p) =>
          p.id === d.id ? { 
            ...p, 
            x: scaleXRef.current.invert(x), 
            y: scaleYRef.current.invert(y) 
          } : p
        );
        
        setOriginalScatterData(updatedOriginalCoords);
        setSelectedPoint({...d, x, y});
      }
      
      function dragended(event, d) {
        // Prevent event from bubbling up to zoom behavior
        event.sourceEvent.stopPropagation();
        
        d3.select(this).attr("stroke", "black");
        // Keep the point selected after drag ends
      }
  
      // Add zoom behavior, preserving the current zoom state
      const zoom = d3.zoom()
        .scaleExtent([0.5, 5]) // Define the zoom scale extent (min, max)
        .on('zoom', (event) => {
          zoomContainer.attr('transform', event.transform); 
          setZoomTransform(event.transform);
        })
        .filter(event => {
          // Disable double-click zooming
          if (event.type === 'dblclick') {
            event.preventDefault();
            return false;
          }
          return !event.button && event.type !== 'dblclick';
        });
  
      // Store the zoom reference for external control
      zoomRef.current = zoom;
      
      // Apply zoom behavior to the SVG
      svg.call(zoom);
      
      // FIX: Always apply the stored transform to preserve zoom state
      if (zoomTransform && zoomTransform !== d3.zoomIdentity) {
        svg.call(zoom.transform, zoomTransform);
      }
    }
  }, [currentImageURL, imageWidth, imageHeight, originalScatterData, needsScaling]);
  
  // FIX: Separate effect for updating point selection styles without re-rendering the entire SVG
  useEffect(() => {
    if (svgRef.current && scatterData.length > 0) {
      const svg = d3.select(svgRef.current);
      
      // Update the visual appearance of circles based on selection
      svg.selectAll('circle')
        .attr('fill', d => {
          // Match circle by data-id attribute
          const id = d.id || d3.select(d).attr('data-id');
          return selectedPoint && id == selectedPoint.id ? 'yellow' : 'red';
        })
        .attr('stroke-width', d => {
          const id = d.id || d3.select(d).attr('data-id');
          return selectedPoint && id == selectedPoint.id ? 2 : 1;
        });
    }
  }, [selectedPoint]);

  // Handle point selection from the table without affecting zoom
  const handlePointSelect = (point) => {
    setSelectedPoint(point);
  };

  const handleScatterData = async () => {
    // If we have multiple images, we'll download data for all of them
    const downloadPromises = [];
    
    for (let i = 0; i < images.length; i++) {
      // Get the original coordinates for this image
      const originalCoords = i === currentImageIndex ? 
        originalScatterData : 
        images[i].originalCoords || images[i].coords;
      
      const payload = {
        coords: originalCoords.map(({id, ...rest}) => rest), // Remove the id field before sending
        name: images[i].name
      };
      
      // Create a promise for each image's data processing
      const processPromise = (async () => {
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
          await downloadTpsFile(payload);
          
          // Also download the annotated image if it exists
          if (result.image_urls && result.image_urls.length > 0) {
            // Convert relative URL to absolute URL using the API base
            const imageUrl = result.image_urls[0].startsWith('http') 
              ? result.image_urls[0] 
              : `${config.apiBaseUrl}${result.image_urls[0].startsWith('/') ? '' : '/'}${result.image_urls[0]}`;
              
            await downloadAnnotatedImage(imageUrl, payload.name);
          } else {
            // If no image URL is provided, we generate a simple visualization
            // using the current image and points data
            await downloadCurrentImageWithPoints(i, payload.name);
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
      setLoading(true);
      const results = await Promise.allSettled(downloadPromises);
      
      // Check for failures
      const failures = results.filter(r => r.status === 'rejected');
      
      if (failures.length > 0) {
        alert(`Some files failed to download: ${failures.length} of ${images.length}`);
      } else {
        alert(`Successfully downloaded ${images.length} TPS file(s) and annotated image(s)`);
      }
    } catch (error) {
      alert('An error occurred during download');
      console.error('Download error:', error);
    } finally {
      setLoading(false);
    }
  };
  
  // Format for displaying coordinates in the table
  const formatCoord = (value) => {
    return value ? Math.round(value * 100) / 100 : 'N/A';
  };
  
  // Function to download TPS file directly to user's downloads
  const downloadTpsFile = async (payload) => {
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
  const downloadAnnotatedImage = async (imageUrl, imageName) => {
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
  const downloadCurrentImageWithPoints = async (imageIndex, imageName) => {
    try {
      // If this is the current image, use the SVG element
      if (imageIndex === currentImageIndex && svgRef.current) {
        // Get the current SVG element
        const svg = svgRef.current;
        
        // Create a canvas to draw the image and points
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
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
          
          scatterData.forEach(point => {
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
        img.src = currentImageURL;
      } else {
        // For non-current images, create a new canvas and use the stored image data
        const img = new Image();
        
        // Create a promise so we can wait for the image to load
        await new Promise((resolve, reject) => {
          img.onload = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas dimensions to match the image
            canvas.width = img.width;
            canvas.height = img.height;
            
            // Draw the background image
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            // Draw each point
            ctx.fillStyle = 'red';
            ctx.strokeStyle = 'black';
            
            images[imageIndex].coords.forEach((point, idx) => {
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
        img.src = images[imageIndex].imageSets.original;
      }
    } catch (error) {
      console.error('Error creating and downloading points overlay image:', error);
      throw error;
    }
  };

  // Handle changing to a different image in the set
  const changeCurrentImage = (index) => {
    if (index < 0 || index >= images.length) return;
    
    // Save current data to the current image before switching
    if (currentImageIndex !== index && images.length > 0) {
      const updatedImages = [...images];
      
      // Save both current scatter data (display) and original coordinates
      updatedImages[currentImageIndex].coords = scatterData;
      updatedImages[currentImageIndex].originalCoords = originalScatterData;
      
      setImages(updatedImages);
    }
    
    // Set new current image
    setCurrentImageIndex(index);
    setImageFilename(images[index].name);
    
    // Set the image sets for the new image
    if (images[index].imageSets) {
      setImageSet(images[index].imageSets);
      setCurrentImageURL(images[index].imageSets.original);
    } else {
      console.error(`Image sets not available for image at index ${index}`);
    }
    
    // We need to reset everything for the new image
    setNeedsScaling(true);
    setSelectedPoint(null);
    
    // Set the scatter data and original data to the new image's coordinates
    setScatterData(images[index].coords);
    setOriginalScatterData(images[index].originalCoords || images[index].coords);
  };

  // Fetch all files in the upload folder on component mount and at regular intervals
  useEffect(() => {
    const fetchUploadedFiles = async () => {
      try {
        // This endpoint needs to be implemented in app.py
        const response = await fetch(`${config.apiBaseUrl}/list_uploads`, {
          method: 'GET',
        });
        
        if (response.ok) {
          const files = await response.json();
          console.log("Files in upload folder:", files);
          
          // Create history entries for files that aren't in current upload history
          const currentFileNames = new Set(uploadHistory.map(item => item.name));
          const newHistory = [...uploadHistory];
          
          files.forEach(file => {
            if (!currentFileNames.has(file)) {
              newHistory.push({
                name: file,
                timestamp: 'From uploads folder',
                index: -1 // Will be set when loaded
              });
            }
          });
          
          if (newHistory.length !== uploadHistory.length) {
            setUploadHistory(newHistory);
          }

          // Update lizard count to include ALL files in the upload folder
          setLizardCount(files.length);
        }
      } catch (error) {
        console.log('Error fetching upload directory files:', error);
        // Even if fetch fails, count unique images in the current session
        const uniqueImages = new Set(images.map(img => img.name));
        setLizardCount(uniqueImages.size);
      }
    };
    
    // Initial fetch
    fetchUploadedFiles();
    
    // Set up interval to check for new files (every 30 seconds)
    const intervalId = setInterval(fetchUploadedFiles, 30000);
    
    // Clean up on unmount
    return () => clearInterval(intervalId);
  }, [uploadHistory, images]);
  
  // Load file from uploads folder when selected from history
  const loadImageFromUploads = async (filename) => {
    try {
      setLoading(true);
      setDataLoading(true);
      
      // Immediately set dataFetched to true to show the UI elements
      setDataFetched(true);
      
      // First check if this file is already loaded in our images array
      const existingIndex = images.findIndex(img => img.name === filename);
      if (existingIndex >= 0) {
        changeCurrentImage(existingIndex);
        setLoading(false);
        setDataLoading(false);
        return;
      }
      
      // If not loaded, we need to process it
      const response = await fetch(`${config.apiBaseUrl}/process_existing?filename=${encodeURIComponent(filename)}`, {
        method: 'POST',
      });
      
      if (response.ok) {
        const result = await response.json();
        const imageSets = await fetchImageSet(filename);
        
        const coords = result.coords.map((coord, index) => ({
          ...coord,
          id: index + 1
        }));
        
        const newImage = {
          name: filename,
          coords: coords,
          originalCoords: JSON.parse(JSON.stringify(coords)),
          imageSets,
          timestamp: 'From uploads folder'
        };
        
        // Add to images array
        const updatedImages = [...images, newImage];
        setImages(updatedImages);
        
        // Set current image data immediately
        setScatterData(coords);
        setOriginalScatterData(JSON.parse(JSON.stringify(coords)));
        setImageSet(imageSets);
        setCurrentImageURL(imageSets.original);
        
        // Update history entry with correct index
        const newHistoryItem = {
          name: filename,
          timestamp: 'From uploads folder',
          index: updatedImages.length - 1
        };
        
        const updatedHistory = uploadHistory.map(item => 
          item.name === filename ? newHistoryItem : item
        );
        
        setUploadHistory(updatedHistory);
        
        // Set as current image
        setCurrentImageIndex(updatedImages.length - 1);
        setImageFilename(filename);
      } else {
        throw new Error('Failed to process existing image');
      }
    } catch (error) {
      console.error('Error loading image from uploads:', error);
      setDataError(error);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveAnnotations = async () => {
    if (!dataFetched || loading || !imageFilename) return;
    
    try {
      setLoading(true);
      
      // Convert scaled coordinates back to original image coordinates
      const originalCoords = scatterData.map(point => ({
        x: scaleXRef.current.invert(point.x),
        y: scaleYRef.current.invert(point.y)
      }));
      
      const payload = {
        coords: originalCoords,
        name: imageFilename
      };
      
      // Send data to backend
      const response = await fetch(`${config.apiBaseUrl}/save_annotations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const result = await response.json();
      
      // Update originalScatterData with the saved coordinates
      setOriginalScatterData(originalCoords.map((coord, index) => ({
        ...coord,
        id: index + 1
      })));
      
      // Update the current image's data in the images array
      const updatedImages = [...images];
      updatedImages[currentImageIndex].originalCoords = originalCoords.map((coord, index) => ({
        ...coord,
        id: index + 1
      }));
      setImages(updatedImages);
      
      // Show success message
      alert('Annotations saved successfully!');
      console.log('Save result:', result);
      
    } catch (error) {
      console.error('Error saving annotations:', error);
      alert(`Error saving annotations: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100vh', 
      minWidth: '1400px' 
    }}>
      {/* Condensed header with height closer to info bar */}
      <div style={{ 
        padding: '15px', 
        borderBottom: '1px solid #ccc', 
        textAlign: 'center',
        minHeight: '180px',
        position: 'relative' 
      }}>
      {/* Info Box - remain at right */}
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '15px',
        padding: '12px',
        backgroundColor: '#f8f9fa',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        border: '1px solid #dee2e6',
        textAlign: 'right',
        minWidth: '300px'
      }}>
        {/* Updated info box content */}
        <div style={{ fontSize: '0.9em', color: '#495057' }}>
          <p style={{ margin: '4px 0' }}>Made with ❤️ by the Human Augmented Analytics Group (HAAG)</p>
          <p style={{ margin: '4px 0' }}>In Partnership with Dr. Stroud</p>
          <p style={{ margin: '4px 0' }}>Author: Mercedes Quintana</p>
          <p style={{ margin: '4px 0' }}>AI Engineer: Anthony Trevino</p>
          <p style={{ margin: '4px 0', fontStyle: 'italic' }}>Georgia Institute of Technology - Spring 2025</p>
          <a href="https://github.com/Human-Augment-Analytics/Lizard-CV-Web-App" 
            target="_blank" 
            rel="noopener noreferrer"
            style={{ color: '#0056b3', textDecoration: 'none' }}>
            View on GitHub
          </a>
          <div style={{ marginTop: '5px', color: '#0056b3' }}>
            <strong>Number of Lizards Analyzed: {lizardCount}</strong>
          </div>
        </div>
      </div>
        
        {/* Main content row with title and buttons horizontally aligned */}
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between',
          alignItems: 'center', 
          maxWidth: '1200px', 
          margin: '15px auto 0', /* Added top margin to move buttons down */
          padding: '0 20px'
        }}>
          {/* Left side - buttons remain left-aligned */}
          <div style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'flex-start', 
            gap: '15px', /* Increased gap between buttons */
            width: '220px',
            marginTop: '20px' /* Added top margin to move buttons down */
          }}>
            {/* Upload button */}
            <label 
              htmlFor="file-upload" 
              style={{
                padding: '12px 20px',
                backgroundColor: '#2196F3',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: loading ? 'not-allowed' : 'pointer',
                display: 'inline-block',
                fontWeight: 'bold',
                fontSize: '14px', /* Standardized font size */
                textAlign: 'center',
                boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                opacity: loading ? 0.7 : 1,
                width: '100%',
                boxSizing: 'border-box'
              }}
            >
              {loading ? 'Uploading...' : 'Upload X-Ray Images'}
            </label>
            <input 
              id="file-upload" 
              type="file" 
              accept="image/*" 
              onChange={handleUpload}
              style={{ display: 'none' }}
              multiple
              disabled={loading}
            />
            
            {/* Export button */}
            <button
              onClick={handleScatterData}
              disabled={!dataFetched || loading}
              style={{
                padding: '12px 20px',
                backgroundColor: '#4CAF50',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: dataFetched && !loading ? 'pointer' : 'not-allowed',
                opacity: loading || !dataFetched ? 0.7 : 1,
                width: '100%',
                boxSizing: 'border-box',
                fontWeight: 'bold',
                fontSize: '14px', /* Standardized font size */
                boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
              }}>
              Export All Data
            </button>
          </div>
          
          {/* Center - title is now horizontally aligned with buttons */}
          <div style={{ 
            display: 'flex', 
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <img 
              src="/android-chrome-192x192.png" 
              alt="Lizard Logo" 
              style={{ height: '40px', marginRight: '12px' }} 
            />
            <h2 style={{ margin: 0, whiteSpace: 'nowrap' }}>Lizard Anolis X-Ray Auto-Annotator</h2>
          </div>
          
          {/* Right spacer - balances the layout */}
          <div style={{ width: '220px' }}></div>
        </div>

        {dataError && <span style={{ color: 'red' }}>Error: {dataError.message}</span>}

        {/* Navigation controls */}
        {images.length > 1 && (
          <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
            <button
              onClick={() => changeCurrentImage(currentImageIndex - 1)}
              disabled={currentImageIndex === 0 || loading}
              style={{
                padding: '8px 12px',
                backgroundColor: '#f0f0f0',
                border: 'none',
                borderRadius: '4px',
                fontSize: '14px',
                cursor: currentImageIndex === 0 || loading ? 'not-allowed' : 'pointer'
              }}
            >
              Previous Image
            </button>
            
            <span>Image {currentImageIndex + 1} of {images.length}</span>
            
            <button
              onClick={() => changeCurrentImage(currentImageIndex + 1)}
              disabled={currentImageIndex === images.length - 1 || loading}
              style={{
                padding: '8px 12px',
                backgroundColor: '#f0f0f0',
                border: 'none',
                borderRadius: '4px',
                fontSize: '14px',
                cursor: currentImageIndex === images.length - 1 || loading ? 'not-allowed' : 'pointer'
              }}
            >
              Next Image
            </button>
          </div>
        )}

        {/* Image version buttons */}
        {dataFetched && imageSet.original && (
          <div style={{ marginTop: '10px', display: 'flex', gap: '10px', justifyContent: 'center' }}>
            <button 
              onClick={() => {
                setNeedsScaling(true);
                setCurrentImageURL(imageSet.original);
              }}
              disabled={loading || dataLoading}
              style={{ 
                padding: '8px 12px',
                backgroundColor: currentImageURL === imageSet.original ? '#2196F3' : '#f0f0f0',
                color: currentImageURL === imageSet.original ? 'white' : 'black',
                border: 'none',
                borderRadius: '4px',
                fontSize: '14px',
                cursor: loading || dataLoading ? 'not-allowed' : 'pointer',
                opacity: loading || dataLoading ? 0.7 : 1
              }}
            >
              Original
            </button>
            <button 
              onClick={() => {
                setNeedsScaling(true);
                setCurrentImageURL(imageSet.inverted);
              }}
              disabled={loading || dataLoading}
              style={{ 
                padding: '8px 12px',
                backgroundColor: currentImageURL === imageSet.inverted ? '#2196F3' : '#f0f0f0',
                color: currentImageURL === imageSet.inverted ? 'white' : 'black',
                border: 'none',
                borderRadius: '4px',
                fontSize: '14px',
                cursor: loading || dataLoading ? 'not-allowed' : 'pointer',
                opacity: loading || dataLoading ? 0.7 : 1
              }}
            >
              Inverted
            </button>
            <button 
              onClick={() => {
                setNeedsScaling(true);
                setCurrentImageURL(imageSet.color_contrasted);
              }}
              disabled={loading || dataLoading}
              style={{ 
                padding: '8px 12px',
                backgroundColor: currentImageURL === imageSet.color_contrasted ? '#2196F3' : '#f0f0f0',
                color: currentImageURL === imageSet.color_contrasted ? 'white' : 'black',
                border: 'none',
                borderRadius: '4px',
                fontSize: '14px',
                cursor: loading || dataLoading ? 'not-allowed' : 'pointer',
                opacity: loading || dataLoading ? 0.7 : 1
              }}
            >
              Color Contrasted
            </button>
          </div>
        )}
      </div>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* NEW: History Table - left side */}
        <div style={{ 
          width: '220px', 
          borderRight: '1px solid #ccc', 
          overflowY: 'auto',
          padding: '10px'
        }}>
          <h3>History</h3>
          <div style={{ 
            maxHeight: 'calc(100vh - 250px)', 
            overflowY: 'auto', 
            border: '1px solid #ddd', 
            borderRadius: '4px'
          }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ backgroundColor: '#f3f3f3' }}>
                  <th style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>Image Name</th>
                </tr>
              </thead>
              <tbody>
                {uploadHistory.length > 0 ? (
                  uploadHistory.map((item, idx) => (
                    <tr 
                      key={idx}
                      onClick={() => item.index >= 0 ? changeCurrentImage(item.index) : loadImageFromUploads(item.name)}
                      style={{
                        cursor: 'pointer',
                        backgroundColor: (item.index === currentImageIndex) ? '#f0f0f0' : 'transparent'
                      }}
                    >
                      <td style={{ 
                        padding: '8px', 
                        borderBottom: '1px solid #eee',
                        fontWeight: (item.index === currentImageIndex) ? 'bold' : 'normal',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}>
                        {item.name}
                        <div style={{ fontSize: '0.8em', color: '#666' }}>{item.timestamp}</div>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td style={{ padding: '10px', textAlign: 'center', color: '#666' }}>
                      No images in history
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
        
        {/* SVG container - center */}
        <div style={{ flex: 3, overflow: 'auto', position: 'relative' }}>
          {!dataFetched && !loading && uploadHistory.length === 0 && (
            <div style={{ 
              position: 'absolute', 
              top: '50%', 
              left: '50%', 
              transform: 'translate(-50%, -50%)',
              textAlign: 'center',
              color: '#666'
            }}>
              <p>Upload one or more X-ray images to begin analysis</p>
              <p style={{ fontSize: '0.9em' }}>The images will appear here</p>
            </div>
          )}
          
          <svg
            ref={svgRef}
            style={{ 
              display: 'block', 
              margin: '0 auto',
              boxShadow: '0 0 5px rgba(0,0,0,0.2)',
              backgroundColor: dataFetched ? '#f9f9f9' : 'transparent'
            }}
          />
          
          {/* Show loading message ONLY if an image has been uploaded and is loading */}
          {dataLoading && dataFetched && (
            <div style={{ 
              position: 'absolute', 
              top: '50%', 
              left: '50%', 
              transform: 'translate(-50%, -50%)',
              backgroundColor: 'rgba(255,255,255,0.8)',
              padding: '15px',
              borderRadius: '5px',
              boxShadow: '0 0 10px rgba(0,0,0,0.1)'
            }}>
              Loading image...
            </div>
          )}
          
          {dataError && !loading && (
            <div style={{ 
              position: 'absolute', 
              top: '50%', 
              left: '50%', 
              transform: 'translate(-50%, -50%)',
              backgroundColor: 'rgba(255,220,220,0.9)',
              padding: '15px',
              borderRadius: '5px',
              color: 'red',
              boxShadow: '0 0 10px rgba(255,0,0,0.2)'
            }}>
              Error: {dataError.message}
            </div>
          )}
        </div>
        
        {/* Points Table - right side */}
        {dataFetched && (
          <div style={{ flex: 1, borderLeft: '1px solid #ccc', padding: '10px', overflowY: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <h3 style={{ margin: 0 }}>Landmark Points</h3>
              <button
                onClick={handleSaveAnnotations}
                disabled={loading}
                style={{
                  padding: '8px 15px',
                  backgroundColor: loading ? '#ccc' : '#ff9800',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontWeight: 'bold',
                  fontSize: '14px',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}
              >
                {loading ? 'Saving...' : 'Save Annotations'}
              </button>
            </div>
            
            {/* Selected Point Details - Moved to top */}
            {selectedPoint && (
              <div style={{ marginBottom: '20px', padding: '10px', backgroundColor: '#f9f9f9', borderRadius: '4px', border: '1px solid #ddd' }}>
                <h4 style={{ marginTop: 0 }}>Selected Point Details</h4>
                <p><strong>Point {selectedPoint.id}</strong></p>
                <p><strong>X coordinate:</strong> {formatCoord(selectedPoint.x)}</p>
                <p><strong>Y coordinate:</strong> {formatCoord(selectedPoint.y)}</p>
                <div style={{ marginTop: '10px', fontSize: '0.9em', color: '#666' }}>
                  <p>Image: {imageFilename}</p>
                  <p>Image {currentImageIndex + 1} of {images.length}</p>
                </div>
              </div>
            )}

            <p>Click on a row to select a point. Selected point is highlighted in yellow.</p>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ backgroundColor: '#f3f3f3' }}>
                  <th style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>Point ID</th>
                  <th style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>X</th>
                  <th style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>Y</th>
                </tr>
              </thead>
              <tbody>
                {scatterData.map((point) => (
                  <tr 
                    key={point.id} 
                    onClick={() => handlePointSelect(point)}
                    style={{ 
                      cursor: 'pointer', 
                      backgroundColor: selectedPoint && selectedPoint.id === point.id ? '#ffff99' : 'transparent',
                      transition: 'background-color 0.2s'
                    }}
                  >
                    <td style={{ padding: '8px', borderBottom: '1px solid #ddd' }}>Point {point.id}</td>
                    <td style={{ padding: '8px', borderBottom: '1px solid #ddd' }}>{formatCoord(point.x)}</td>
                    <td style={{ padding: '8px', borderBottom: '1px solid #ddd' }}>{formatCoord(point.y)}</td>
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

export default App;