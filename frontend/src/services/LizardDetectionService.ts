export interface LizardDetection {
  bbox: number[]; // [x, y, width, height]
  class: string;
  score: number;
}

import { OnnxDetectionService } from './OnnxDetectionService';

export class LizardDetectionService {
  private static readonly API_BASE = 'http://localhost:3005/api';
  private static isInitialized = false;

  /**
   * Initialize the ONNX model
   */
  static async initialize(): Promise<void> {
    if (!this.isInitialized) {
      await OnnxDetectionService.initialize('/models/best.onnx', ['finger', 'toe', 'ruler']);
      this.isInitialized = true;
    }
  }

  /**
   * Detect lizard toepads in an image using Python backend (default method)
   */
  static async detectLizardToepads(
    imageFile: File,
    confidenceThreshold: number = 0.1
  ): Promise<LizardDetection[]> {
    return this.detectLizardToepadsPython(imageFile, confidenceThreshold);
  }

  /**
   * Detect lizard toepads using Python preprocessing + ONNX inference (hybrid method)
   */
  static async detectLizardToepadsOnnx(
    imageFile: File,
    confidenceThreshold: number = 0.1
  ): Promise<LizardDetection[]> {
    try {
      // Step 1: Use Python backend for preprocessing (to match YOLO pipeline exactly)
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('confidence', confidenceThreshold.toString());
      formData.append('preprocess_only', 'true'); // Flag to only do preprocessing

      const response = await fetch(`${this.API_BASE}/preprocess-image`, {
        method: 'POST',
        body: formData,
        mode: 'cors',
        headers: {
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Preprocessing failed: ${response.statusText}`);
      }

      const preprocessResult = await response.json();
      
      // Step 2: Use ONNX for model inference with preprocessed image
      await this.initialize();

      // Create image element from preprocessed image data
      const processedImageData = preprocessResult.processed_image; // Base64 or image data
      const img = new Image();
      
      return new Promise((resolve, reject) => {
        img.onload = async () => {
          try {
            // Run ONNX detection on preprocessed image
            // The preprocessed image is already in the correct format (1024x489)
            // So we need to use a different preprocessing method that doesn't do letterbox
            const detections = await OnnxDetectionService.detectObjectsDirectResize(
              img,
              confidenceThreshold,
              0.45, // IoU threshold
              preprocessResult.processed_width,
              preprocessResult.processed_height
            );
            
            // Convert coordinates back to original image scale using the scale factor
            const scaleX = preprocessResult.original_width / preprocessResult.processed_width;
            const scaleY = preprocessResult.original_height / preprocessResult.processed_height;
            
            // Convert to our format and scale coordinates
            const lizardDetections: LizardDetection[] = detections.map(det => ({
              bbox: [
                det.bbox[0] * scaleX, // x
                det.bbox[1] * scaleY, // y  
                det.bbox[2] * scaleX, // width
                det.bbox[3] * scaleY  // height
              ],
              class: det.class.toLowerCase(),
              score: det.score
            }));
            
            resolve(lizardDetections);
          } catch (error) {
            reject(error);
          }
        };
        
        img.onerror = () => {
          reject(new Error('Failed to load preprocessed image'));
        };
        
        img.src = `data:image/jpeg;base64,${processedImageData}`;
      });
    } catch (error) {
      console.error('Hybrid detection error:', error);
      throw new Error(`Failed to detect lizard toepads: ${error}`);
    }
  }

  /**
   * Detect lizard toepads using Python backend (default method)
   */
  static async detectLizardToepadsPython(
    imageFile: File,
    confidenceThreshold: number = 0.1
  ): Promise<LizardDetection[]> {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('confidence', confidenceThreshold.toString());

    try {
      const response = await fetch(`${this.API_BASE}/detect-lizard`, {
        method: 'POST',
        body: formData,
        mode: 'cors',
        headers: {
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Detection failed: ${response.statusText}`);
      }

      const result = await response.json();
      return result.detections || [];
    } catch (error) {
      console.error('Lizard detection error:', error);
      throw new Error(`Failed to detect lizard toepads: ${error}`);
    }
  }

  /**
   * Get class names for lizard detection
   */
  static getClassNames(): string[] {
    return ['finger', 'toe', 'ruler'];
  }

  /**
   * Get color for each class (matching Lizard_Toepads approach)
   */
  static getClassColor(className: string): string {
    const colors: Record<string, string> = {
      'finger': '#0000FF', // Blue
      'toe': '#FF0000',    // Red
      'ruler': '#800080'   // Purple
    };
    return colors[className.toLowerCase()] || '#00FF00';
  }
}

