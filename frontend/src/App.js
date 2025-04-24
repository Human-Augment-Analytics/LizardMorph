import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';

function App() {
  const [imageUrl, setImageUrl] = useState(null);
  const [needsScaling, setNeedsScaling] = useState(true)
  const [currentImageURL, setCurrentImageURL] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imageWidth, setImageWidth] = useState(0);
  const [imageHeight, setImageHeight] = useState(0);
  const [scatterData, setScatterData] = useState([]);
  const [downloadData, setdownloadData] = useState([]);
  const [dataError, setDataError] = useState(null);
  const [dataLoading, setDataLoading] = useState(true);
  const svgRef = useRef(null);
  const [imageFilename, setImageFilename] = useState(null);
  const [dataFetched, setDataFetched] = useState(false)
  const scaleXRef = useRef(null);
  const scaleYRef = useRef(null);
  const [selectedImageVersion, setSelectedImageVersion] = useState('original');
  
  const [imageSet, setImageSet] = useState({
    original: null,
    inverted: null,
    color_contrasted: null
  });

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      const formData = new FormData();
      formData.append('image', file);

      setLoading(true);

      try {
        const response = await fetch('http://localhost:5000/data', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          setImageFilename(result.name);  
          console.log(result.name)
          setScatterData(result.coords)
          setDataFetched(true)
          console.log(result)
        } else {
          const errorResult = await response.json();
          console.error('Error:', errorResult.error);
        }
      } catch (err) {
        console.error('Error:', err);
      } finally {
        setLoading(false);
      }
    }
  };

  
  const fetchImageSet = async () => {
    try {
      const response = await fetch(`http://localhost:5000/image?image_filename=${encodeURIComponent(imageFilename)}`, {
        method: 'POST',
        headers: {
          'Access-Control-Allow-Origin': '*',
        }
      });
  
      const result = await response.json();
  
      if (result.error) {
        console.error(result.error);
      } else {
        setImageSet({
          original: `data:image/jpeg;base64,${result.image3}`,
          inverted: `data:image/jpeg;base64,${result.image2}`,
          color_contrasted: `data:image/jpeg;base64,${result.image1}`
        });
        setCurrentImageURL(`data:image/jpeg;base64,${result.image3}`);
      }
    } catch (err) {
      console.error('Error fetching image set:', err);
    }
  };
  
  useEffect(() => {
    if (imageFilename) {
      fetchImageSet();
    }
  }, [imageFilename]);

  useEffect(() => {
    if (imageFilename) {
      const fetchImageData = async () => {
        try {
          // ?image_filename=${encodeURIComponent(imageFilename)}
          const response = await fetch(`http://localhost:5000/image?image_filename=${encodeURIComponent(imageFilename)}`, {
            method: 'POST',
            headers: {
              'Access-Control-Allow-Origin': '*'
            }
          });

          const blob = await response.blob();
          const imageUrl = URL.createObjectURL(blob);

          console.log("Fetched image URL: ", imageUrl);
          setImageUrl(imageUrl);  
          setCurrentImageURL(imageUrl)
        } catch (error) {
          setDataError(error);  
        } finally {
          setDataLoading(false); 
        }
      };

      fetchImageData();
    }
  }, [imageFilename]);

  useEffect(() => {
    if (currentImageURL) {
      const img = new Image();
      img.src = currentImageURL
      img.onload = () => {
        setImageWidth(img.width);
        setImageHeight(img.height);
      img.onerror = () => {
        setDataError(new Error('Failed to load image.'));
        setDataLoading(false);
        };
        
      };
      img.src = currentImageURL;
    }
  }, [currentImageURL]);

    useEffect(() => {
      if (currentImageURL && imageWidth && imageHeight) {

        const svg = d3.select(svgRef.current);
        svg.select('image.background-img').remove();
        const windowHeight = window.innerHeight - window.innerHeight*.2;
        const width = windowHeight * (imageWidth / imageHeight);
        const height = windowHeight;
        // const width = imageWidth;
        // const height = imageHeight;
        svg.attr('width', width).attr('height', height);
        const xScale = width / imageWidth;
        const yScale = height / imageHeight;


        scaleXRef.current = d3.scaleLinear()
        .domain([d3.min(scatterData, d => d.x), d3.max(scatterData, d => d.x)])
        .range([d3.min(scatterData, d => d.x)*xScale, d3.max(scatterData, d => d.x)*xScale])
        scaleYRef.current = d3.scaleLinear()
        .domain([d3.min(scatterData, d => d.y), d3.max(scatterData, d => d.y)])
        .range([d3.min(scatterData, d => d.y)*yScale,d3.max(scatterData, d => d.y)*yScale ])
        if (needsScaling){
          const scaledData = scatterData.map(point => ({
            x: scaleXRef.current(point.x),
            y: scaleYRef.current(point.y),
          }));
          setScatterData(scaledData);
          setNeedsScaling(false)
        }
        const image = svg
          .append('image')
          .attr('class', 'background-img')
          .attr('href', currentImageURL)  
          .attr('x', 0)            
          .attr('y', 0)          
          .attr('width', width)    
          .attr('height', height) 
          .attr('preserveAspectRatio', 'xMidYMid slice') 

        const scatterPlotGroup = svg.append('g').attr('class', 'scatter-points');
  
  scatterPlotGroup
    .selectAll('circle')
    .data(scatterData)
    .enter()
    .append('circle')
    // .attr('cx', (d) => d.x * yScale) // x-coordinate of the points
    .attr('cx', (d) => d.x)
    //.attr('cy', (d) => height - (d.y * xScale)) // Correct positioning with bottom-left origin
    .attr('cy', (d) => d.y)
    .attr('r', 2)  // Radius of each scatter plot point
    .attr('fill', 'red')  // Color of the points
    .attr('stroke', 'black') // Optional stroke color for visibility
    .call(d3.drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended));
  
    function dragstarted(event, d) {
      d3.select(this).raise().attr("stroke", "black");
    }
    
    function dragged(event, d) {
      const newX = event.x;  // Convert to image space
      const newY = event.y; 
  
      d3.select(this)
        // .attr("cx", d.x = scaleX(event.x))
        // .attr("cy", d.y = scaleY(event.y));
        .attr("cx", d.x = event.x)
        .attr("cy", d.y = event.y);
  
        // const boundedX = Math.max(0, Math.min(imageWidth, newX));
        // const boundedY = Math.max(0, Math.min(imageHeight, newY));
  
        // d3.select(this)
        // .attr("cx", boundedX * xScale) // Convert back to SVG space
        // .attr("cy", boundedY * yScale);
  
        const updatedScatterData = scatterData.map((point) =>
          point === d ? { ...point, x: event.x, y: event.y } : point
        );
        setScatterData(updatedScatterData);
    }
    
    function dragended(event, d) {
      d3.select(this).attr("stroke", null);
    }
        // scatterPlotRef.current = scatterPlotGroup; // Reference to scatter plot group
  
        // Add zoom behavior
        const zoom = d3.zoom()
          .scaleExtent([0.5, 5]) // Define the zoom scale extent (min, max)
          .on('zoom', (event) => {
            svg.selectAll('image, .scatter-points')
              .attr('transform', event.transform); // Apply zoom transformation to both image and scatter points
          });
  
        svg.call(zoom);
      }
    }, [currentImageURL, imageWidth, imageHeight]); 

    const handleScatterData = async () => {
      const temp = scatterData.map(point => ({ ...point }));
      for (let i = 0; i < scatterData.length; i++){
        temp[i].x = scaleXRef.current.invert(scatterData[i].x)
        temp[i].y = scaleYRef.current.invert(scatterData[i].y)
      }
      const payload = {
        coords: temp,
        name: imageFilename
      }
      try {
        const response = await fetch('http://localhost:5000/endpoint', {
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
        console.log(result);
      } catch (error) {
        // Handle any errors that occur during the fetch request
        console.error("Error during the fetch request:", error);
      }

    }

  return (
    <div>
    <div style={{ position: 'relative' }}>
      <h2>Upload an Image</h2>
      <input type="file" accept="image/*" onChange={handleUpload}/>
      <button
      onClick={handleScatterData}
      style={{visibility: dataFetched ? 'visible ' : 'hidden'}}>
        Download
      </button>
      
    </div>
    <div style={{ marginTop: '10px' }}>
      <button onClick={() => setCurrentImageURL(imageSet.original)} style={{ visibility: dataFetched ? 'visible' : 'hidden' }}>Original</button>
      <button onClick={() => setCurrentImageURL(imageSet.inverted)}style={{ visibility: dataFetched ? 'visible' : 'hidden' }}>Inverted</button>
      <button onClick={() => setCurrentImageURL(imageSet.color_contrasted)}style={{ visibility: dataFetched ? 'visible' : 'hidden' }}>Color Contrasted</button>
    </div>

    <div>
      <svg
        ref={svgRef}
        style={{ display: 'block', marginTop: '20px'}}
      >
      </svg>
      </div>   
      </div>
   
  );
}

export default App;





