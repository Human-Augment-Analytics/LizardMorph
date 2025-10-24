import React, { useState, useRef } from 'react';
import { LizardDetectionService, type LizardDetection } from '../services/LizardDetectionService';

export const LizardDetectionPage: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detections, setDetections] = useState<LizardDetection[]>([]);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.1);
  const [useOnnx, setUseOnnx] = useState(true); // Toggle between ONNX (hybrid) and Python (Python default)

  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    if (!fileInputRef.current?.files?.[0]) return;

    try {
      setIsLoading(true);
      setError(null);

      const startTime = performance.now();
      
      let results: LizardDetection[];
      if (useOnnx) {
        // Use ONNX detection
        results = await LizardDetectionService.detectLizardToepadsOnnx(
          fileInputRef.current.files[0],
          confidenceThreshold
        );
      } else {
        // Use Python backend detection (default)
        results = await LizardDetectionService.detectLizardToepads(
          fileInputRef.current.files[0],
          confidenceThreshold
        );
      }
      
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

  const drawDetections = (detections: LizardDetection[]) => {
    if (!canvasRef.current || !imageRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = imageRef.current;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    // Draw image
    ctx.drawImage(img, 0, 0);

    // Draw detections with class-specific colors
    detections.forEach((detection, index) => {
      const [x, y, width, height] = detection.bbox;
      const color = LizardDetectionService.getClassColor(detection.class);
      
      // For ONNX, coordinates are already in original image scale
      // For Python backend, we need to scale from processed image
      let scaledX = x, scaledY = y, scaledWidth = width, scaledHeight = height;
      
      if (!useOnnx) {
        // Scale coordinates from processed image (1024x489) to original image size
        const processedWidth = 1024;
        const processedHeight = 489;
        const scaleX = img.naturalWidth / processedWidth;
        const scaleY = img.naturalHeight / processedHeight;
        
        scaledX = x * scaleX;
        scaledY = y * scaleY;
        scaledWidth = width * scaleX;
        scaledHeight = height * scaleY;
      }

      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 6;
      ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

      // Draw label background
      const label = `${detection.class} ${(detection.score * 100).toFixed(1)}%`;
      ctx.font = '16px Arial';
      const textMetrics = ctx.measureText(label);
      const textHeight = 20;

      ctx.fillStyle = color;
      ctx.fillRect(scaledX, scaledY - textHeight, textMetrics.width + 8, textHeight);

      // Draw label text
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(label, scaledX + 4, scaledY - 5);
    });
  };

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto', color: '#e0e0e0' }}>
      <h1 style={{ color: '#ffffff' }}>Lizard Toepad Detection</h1>

      {/* Detection Method Toggle */}
      <div style={{
        padding: '10px',
        marginBottom: '20px',
        backgroundColor: '#2a2a2a',
        border: '1px solid #404040',
        borderRadius: '4px',
        color: '#ffffff'
      }}>
        <strong>Detection Method: </strong>
        <label style={{ marginLeft: '10px', marginRight: '20px' }}>
          <input
            type="radio"
            name="detectionMethod"
            checked={useOnnx}
            onChange={() => setUseOnnx(true)}
            style={{ marginRight: '5px' }}
          />
          ONNX (Browser-based)
        </label>
        <label>
          <input
            type="radio"
            name="detectionMethod"
            checked={!useOnnx}
            onChange={() => setUseOnnx(false)}
            style={{ marginRight: '5px' }}
          />
          Python Backend (YOLO)
        </label>
      </div>

      {/* Status */}
      <div style={{
        padding: '10px',
        marginBottom: '20px',
        backgroundColor: '#1e4620',
        border: '1px solid #2d5f2f',
        borderRadius: '4px',
        color: '#ffffff'
      }}>
        <strong>Detection Service: </strong>
        {useOnnx ? 'ONNX (Hybrid) Ready ✓' : 'Python Backend (YOLO) Ready ✓'}
        <div style={{ marginTop: '10px' }}>
          <strong>Available Classes: </strong>
          {LizardDetectionService.getClassNames().join(', ')}
        </div>
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
              Confidence Threshold: {confidenceThreshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              style={{ width: '200px' }}
            />
          </div>
          <button
            onClick={handleDetect}
            disabled={isLoading}
            style={{
              padding: '10px 20px',
              backgroundColor: !isLoading ? '#007bff' : '#6c757d',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: !isLoading ? 'pointer' : 'not-allowed',
              fontSize: '16px'
            }}
          >
            {isLoading ? 'Detecting...' : `Detect Lizard Toepads (${useOnnx ? 'ONNX Hybrid' : 'Python YOLO'})`}
          </button>
        </div>
      )}

      {/* Results */}
      {processingTime !== null && (
        <div style={{ marginBottom: '20px', color: '#ffffff' }}>
          <strong>Processing Time: </strong>{processingTime.toFixed(2)} ms
          <br />
          <strong>Detections: </strong>{detections.length} objects found
          <br />
          <strong>Method: </strong>{useOnnx ? 'ONNX (Hybrid - Python preprocessing + ONNX inference)' : 'Python Backend (YOLO)'}
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
                <th style={{ padding: '8px', border: '1px solid #555', color: '#ffffff' }}>Color</th>
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
                  <td style={{ padding: '8px', border: '1px solid #555' }}>
                    <div style={{
                      width: '20px',
                      height: '20px',
                      backgroundColor: LizardDetectionService.getClassColor(det.class),
                      border: '1px solid #555'
                    }}></div>
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
