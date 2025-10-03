import React, { useState, useRef, useEffect } from 'react';
import { OnnxDetectionService } from '../services/OnnxDetectionService';
import type { Detection } from '../services/OnnxDetectionService';

export const YoloTestPage: React.FC = () => {
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [scoreThreshold, setScoreThreshold] = useState(0.5);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [modelUrl, setModelUrl] = useState('/models/yolov5n-seg.onnx');

  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleLoadModel = async () => {
    try {
      setIsLoading(true);
      setError(null);
      await OnnxDetectionService.initialize(modelUrl);
      setIsModelLoaded(true);
    } catch (err) {
      setError(`Failed to initialize model: ${err}`);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      OnnxDetectionService.dispose();
    };
  }, []);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      setImageUrl(e.target?.result as string);
      setDetections([]);
      setProcessingTime(null);
    };
    reader.readAsDataURL(file);
  };

  const handleDetect = async () => {
    if (!imageRef.current || !isModelLoaded) return;

    try {
      setIsLoading(true);
      setError(null);

      const startTime = performance.now();
      const results = await OnnxDetectionService.detectObjects(
        imageRef.current,
        scoreThreshold,
        iouThreshold
      );
      const endTime = performance.now();

      setDetections(results);
      setProcessingTime(endTime - startTime);
      drawDetections(results);
    } catch (err) {
      setError(`Detection failed: ${err}`);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const drawDetections = (detections: Detection[]) => {
    if (!canvasRef.current || !imageRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = imageRef.current;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    // Draw image
    ctx.drawImage(img, 0, 0);

    // Draw detections
    detections.forEach((detection) => {
      const [x, y, width, height] = detection.bbox;

      // Draw bounding box
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      // Draw label background
      const label = `${detection.class} ${(detection.score * 100).toFixed(1)}%`;
      ctx.font = '16px Arial';
      const textMetrics = ctx.measureText(label);
      const textHeight = 20;

      ctx.fillStyle = '#00FF00';
      ctx.fillRect(x, y - textHeight, textMetrics.width + 8, textHeight);

      // Draw label text
      ctx.fillStyle = '#000000';
      ctx.fillText(label, x + 4, y - 5);
    });
  };

  // Memory info not available for ONNX Runtime
  // const memoryInfo = { numTensors: 0, numBytes: 0 };

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto', color: '#e0e0e0' }}>
      <h1 style={{ color: '#ffffff' }}>YOLOv5 Object Detection Test</h1>

      {/* Model URL input */}
      {!isModelLoaded && (
        <div style={{ marginBottom: '20px' }}>
          <div style={{ marginBottom: '10px' }}>
            <label style={{ display: 'block', marginBottom: '5px' }}>
              <strong>Model URL:</strong>
            </label>
            <input
              type="text"
              value={modelUrl}
              onChange={(e) => setModelUrl(e.target.value)}
              placeholder="Path to model.json"
              style={{
                width: '400px',
                padding: '8px',
                marginRight: '10px',
                border: '1px solid #ccc',
                borderRadius: '4px'
              }}
            />
            <button
              onClick={handleLoadModel}
              disabled={isLoading}
              style={{
                padding: '8px 16px',
                backgroundColor: isLoading ? '#6c757d' : '#28a745',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: isLoading ? 'not-allowed' : 'pointer'
              }}
            >
              {isLoading ? 'Loading Model...' : 'Load Model'}
            </button>
          </div>
          <small style={{ color: '#a0a0a0' }}>
            Enter the URL to your ONNX model file (.onnx)
            <br />
            <strong>Default:</strong> /models/yolov5n-seg.onnx
          </small>
        </div>
      )}

      {/* Status */}
      <div style={{
        padding: '10px',
        marginBottom: '20px',
        backgroundColor: isModelLoaded ? '#1e4620' : '#2a2a2a',
        border: `1px solid ${isModelLoaded ? '#2d5f2f' : '#404040'}`,
        borderRadius: '4px',
        color: '#ffffff'
      }}>
        <strong>Model Status: </strong>
        {isLoading ? 'Loading...' : isModelLoaded ? 'Ready ✓' : 'Not Loaded'}
      </div>

      {/* Error display */}
      {error && (
        <div style={{
          padding: '10px',
          marginBottom: '20px',
          backgroundColor: '#4a2020',
          border: '1px solid #6a3030',
          borderRadius: '4px',
          color: '#ff9999'
        }}>
          <strong>Error: </strong>{error}
        </div>
      )}

      {/* File input */}
      <div style={{ marginBottom: '20px' }}>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          style={{ marginRight: '10px' }}
        />
      </div>

      {/* Detection controls */}
      {imageUrl && (
        <div style={{ marginBottom: '20px' }}>
          <div style={{ marginBottom: '10px' }}>
            <label style={{ marginRight: '10px', color: '#e0e0e0' }}>
              Score Threshold: {scoreThreshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={scoreThreshold}
              onChange={(e) => setScoreThreshold(parseFloat(e.target.value))}
              style={{ width: '200px' }}
            />
          </div>
          <div style={{ marginBottom: '10px' }}>
            <label style={{ marginRight: '10px', color: '#e0e0e0' }}>
              IoU Threshold: {iouThreshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={iouThreshold}
              onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
              style={{ width: '200px' }}
            />
          </div>
          <button
            onClick={handleDetect}
            disabled={!isModelLoaded || isLoading}
            style={{
              padding: '10px 20px',
              backgroundColor: isModelLoaded && !isLoading ? '#007bff' : '#6c757d',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: isModelLoaded && !isLoading ? 'pointer' : 'not-allowed',
              fontSize: '16px'
            }}
          >
            {isLoading ? 'Detecting...' : 'Detect Objects'}
          </button>
        </div>
      )}

      {/* Results */}
      {processingTime !== null && (
        <div style={{ marginBottom: '20px', color: '#ffffff' }}>
          <strong>Processing Time: </strong>{processingTime.toFixed(2)} ms
          <br />
          <strong>Detections: </strong>{detections.length} objects found
        </div>
      )}

      {/* Image display */}
      <div style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
        {imageUrl && (
          <>
            <div>
              <h3 style={{ color: '#ffffff' }}>Original Image</h3>
              <img
                ref={imageRef}
                src={imageUrl}
                alt="Input"
                style={{ maxWidth: '500px', border: '1px solid #555' }}
                onLoad={() => {
                  if (canvasRef.current && imageRef.current) {
                    const canvas = canvasRef.current;
                    canvas.width = imageRef.current.naturalWidth;
                    canvas.height = imageRef.current.naturalHeight;
                  }
                }}
              />
            </div>
            <div>
              <h3 style={{ color: '#ffffff' }}>Detections</h3>
              <canvas
                ref={canvasRef}
                style={{ maxWidth: '500px', border: '1px solid #555' }}
              />
            </div>
          </>
        )}
      </div>

      {/* Detection list */}
      {detections.length > 0 && (
        <div>
          <h3 style={{ color: '#ffffff' }}>Detected Objects</h3>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            border: '1px solid #555'
          }}>
            <thead>
              <tr style={{ backgroundColor: '#2a2a2a' }}>
                <th style={{ padding: '8px', border: '1px solid #555', color: '#ffffff' }}>Class</th>
                <th style={{ padding: '8px', border: '1px solid #555', color: '#ffffff' }}>Confidence</th>
                <th style={{ padding: '8px', border: '1px solid #555', color: '#ffffff' }}>Bounding Box</th>
              </tr>
            </thead>
            <tbody>
              {detections.map((det, idx) => (
                <tr key={idx} style={{ backgroundColor: idx % 2 === 0 ? '#1a1a1a' : '#222222' }}>
                  <td style={{ padding: '8px', border: '1px solid #555', color: '#e0e0e0' }}>{det.class}</td>
                  <td style={{ padding: '8px', border: '1px solid #555', color: '#e0e0e0' }}>
                    {(det.score * 100).toFixed(2)}%
                  </td>
                  <td style={{ padding: '8px', border: '1px solid #555', color: '#e0e0e0' }}>
                    [{det.bbox.map(v => v.toFixed(1)).join(', ')}]
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

    </div>
  );
};
 
