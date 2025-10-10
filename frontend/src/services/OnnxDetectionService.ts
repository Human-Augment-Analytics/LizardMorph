import * as ort from 'onnxruntime-web';

export interface Detection {
  bbox: number[]; // [x, y, width, height]
  class: string;
  score: number;
  mask?: ImageData; // For segmentation models
}

export class OnnxDetectionService {
  private static session: ort.InferenceSession | null = null;
  private static isInitialized = false;
  private static isInitializing = false;
  private static classNames: string[] = [];

  // YOLOv5 model configuration
  private static readonly MODEL_INPUT_SIZE = 640;
  // Lizard body part detection model - multiple classes
  private static readonly DEFAULT_CLASS_NAMES = ['Finger', 'Toe', 'Ruler'];

  /**
   * Initialize ONNX Runtime and load the model
   * @param modelUrl - URL to the ONNX model
   * @param customClassNames - Optional custom class names array
   */
  static async initialize(modelUrl?: string, customClassNames?: string[]): Promise<void> {
    if (this.isInitialized) {
      console.log('ONNX model already initialized');
      return;
    }

    if (this.isInitializing) {
      console.log('ONNX model is already initializing...');
      while (this.isInitializing) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      return;
    }

    try {
      this.isInitializing = true;
      console.log('Initializing ONNX Runtime...');

      // Configure ONNX Runtime environment - use minimal configuration
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = false;
      ort.env.wasm.proxy = false;
      
      // Set WASM paths - ensure proper path resolution
      if (typeof window !== 'undefined') {
        const baseUrl = window.location.origin;
        ort.env.wasm.wasmPaths = baseUrl + '/';
        console.log('WASM paths set to:', baseUrl + '/');
      }

      const defaultModelUrl = modelUrl || '/models/best.onnx';
      console.log(`Loading ONNX model from ${defaultModelUrl}...`);

      // Try different execution provider configurations in order of preference
      const executionProviders = ['webgl', 'wasm', 'cpu'] as const;
      const sessionOptions = {
        graphOptimizationLevel: 'all' as const,
        enableCpuMemArena: false,
        enableMemPattern: false,
        enableMemReuse: false,
      };

      let sessionCreated = false;
      for (const provider of executionProviders) {
        try {
          const options = { ...sessionOptions, executionProviders: [provider] };
          this.session = await ort.InferenceSession.create(defaultModelUrl, options);
          console.log(`Successfully loaded model with ${provider} backend`);
          sessionCreated = true;
          break;
        } catch (error) {
          console.warn(`${provider} execution provider failed:`, error);
          if (provider === 'webgl') {
            // Try WASM with different configurations
            try {
              ort.env.wasm.simd = true;
              const wasmOptions = { ...sessionOptions, executionProviders: ['wasm'] as const };
              this.session = await ort.InferenceSession.create(defaultModelUrl, wasmOptions);
              console.log('Successfully loaded model with SIMD WASM backend');
              sessionCreated = true;
              break;
            } catch (wasmError) {
              console.warn('SIMD WASM also failed:', wasmError);
            }
          }
        }
      }

      if (!sessionCreated || !this.session) {
        throw new Error('Failed to initialize model with any execution provider');
      }

      console.log('Model inputs:', this.session.inputNames);
      console.log('Model outputs:', this.session.outputNames);

      // Set class names
      this.classNames = customClassNames || this.DEFAULT_CLASS_NAMES;
      console.log(`Using ${this.classNames.length} class names:`, this.classNames);

      this.isInitialized = true;
      console.log('ONNX model loaded and ready');
    } catch (error) {
      console.error('Failed to initialize ONNX model:', error);
      throw new Error(`Failed to load ONNX model: ${error}`);
    } finally {
      this.isInitializing = false;
    }
  }

  /**
   * Check if the model is initialized
   */
  static isReady(): boolean {
    return this.isInitialized && this.session !== null;
  }

  /**
   * Get all available class names
   */
  static getClassNames(): string[] {
    return [...this.classNames];
  }

  /**
   * Preprocess image to model input format with letterboxing
   */
  private static preprocessImage(imageElement: HTMLImageElement | HTMLCanvasElement): {
    tensor: ort.Tensor;
    originalWidth: number;
    originalHeight: number;
    padX: number;
    padY: number;
    scale: number;
  } {
    // Create canvas for image processing
    const canvas = document.createElement('canvas');
    canvas.width = this.MODEL_INPUT_SIZE;
    canvas.height = this.MODEL_INPUT_SIZE;
    const ctx = canvas.getContext('2d')!;

    // Get original dimensions
    const originalWidth = imageElement.width || (imageElement as HTMLImageElement).naturalWidth;
    const originalHeight = imageElement.height || (imageElement as HTMLImageElement).naturalHeight;

    // Calculate scaling factor (letterboxing - maintain aspect ratio)
    const scale = Math.min(
      this.MODEL_INPUT_SIZE / originalWidth,
      this.MODEL_INPUT_SIZE / originalHeight
    );
    
    const scaledWidth = originalWidth * scale;
    const scaledHeight = originalHeight * scale;
    
    // Calculate padding to center the image
    const padX = (this.MODEL_INPUT_SIZE - scaledWidth) / 2;
    const padY = (this.MODEL_INPUT_SIZE - scaledHeight) / 2;

    // Fill with gray background (typical for YOLOv5 letterboxing)
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, this.MODEL_INPUT_SIZE, this.MODEL_INPUT_SIZE);

    // Draw letterboxed image
    ctx.drawImage(imageElement, padX, padY, scaledWidth, scaledHeight);

    // Get image data
    const imageData = ctx.getImageData(0, 0, this.MODEL_INPUT_SIZE, this.MODEL_INPUT_SIZE);
    const pixels = imageData.data;

    // Convert to grayscale (matching training preprocessing)
    // Formula: grayscale = 0.299*R + 0.587*G + 0.114*B
    const grayscale: number[] = [];
    for (let i = 0; i < pixels.length; i += 4) {
      const gray = (0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2]) / 255.0;
      grayscale.push(gray);
    }

    // Replicate grayscale to 3 channels for RGB model input
    const input = Float32Array.from([...grayscale, ...grayscale, ...grayscale]);
    const tensor = new ort.Tensor('float32', input, [1, 3, this.MODEL_INPUT_SIZE, this.MODEL_INPUT_SIZE]);

    console.log('Letterbox preprocessing:', {
      originalSize: `${originalWidth}x${originalHeight}`,
      scale: scale.toFixed(3),
      scaledSize: `${scaledWidth.toFixed(1)}x${scaledHeight.toFixed(1)}`,
      padding: `x=${padX.toFixed(1)}, y=${padY.toFixed(1)}`,
      imageContentArea: `y=${padY.toFixed(1)} to y=${(padY + scaledHeight).toFixed(1)}`,
      colorMode: 'grayscale (replicated to 3 channels)'
    });

    return { tensor, originalWidth, originalHeight, padX, padY, scale };
  }

  /**
   * Detect objects in an image
   * @param imageElement - HTML image element or canvas
   * @param scoreThreshold - Minimum confidence score (0-1)
   * @param iouThreshold - IoU threshold for NMS
   * @returns Array of detections
   */
  static async detectObjects(
    imageElement: HTMLImageElement | HTMLCanvasElement,
    scoreThreshold = 0.25, // Lowered default threshold for better detection
    iouThreshold = 0.45
  ): Promise<Detection[]> {
    if (!this.isReady()) {
      throw new Error('ONNX model not initialized. Call initialize() first.');
    }

    try {
      // Preprocess image
      const { tensor, originalWidth, originalHeight, padX, padY, scale } = this.preprocessImage(imageElement);

      // Run inference
      const feeds: Record<string, ort.Tensor> = {};
      feeds[this.session!.inputNames[0]] = tensor;

      const results = await this.session!.run(feeds);

      // Get output tensor (YOLOv5 output format)
      const output = results[this.session!.outputNames[0]];

      // Process detections
      const detections = this.processOutput(
        output,
        originalWidth,
        originalHeight,
        padX,
        padY,
        scale,
        scoreThreshold,
        iouThreshold
      );

      return detections;
    } catch (error) {
      console.error('Detection error:', error);
      throw error;
    }
  }

  /**
   * Process YOLO output and apply NMS
   */
  private static processOutput(
    output: ort.Tensor,
    originalWidth: number,
    originalHeight: number,
    padX: number,
    padY: number,
    scale: number,
    scoreThreshold: number,
    iouThreshold: number
  ): Detection[] {
    const data = output.data as Float32Array;
    const dims = output.dims;

    // YOLOv5 output format can be either:
    // [batch, num_detections, values_per_detection] or [batch, values_per_detection, num_detections]
    // We need to detect which format and handle accordingly
    let numDetections: number;
    let valuesPerDetection: number;
    
    // If dims[1] is small (like 6-10), it's likely [batch, values, detections] - transposed
    if (dims[1] < dims[2]) {
      // Transposed format: [1, values, detections]
      valuesPerDetection = dims[1];
      numDetections = dims[2];
      console.log('Detected transposed output format');
    } else {
      // Standard format: [1, detections, values]
      numDetections = dims[1];
      valuesPerDetection = dims[2];
      console.log('Detected standard output format');
    }
    
    const numClasses = valuesPerDetection - 5; // 5 = 4 bbox coords + 1 objectness

    console.log(`Processing ${numDetections} detections with ${valuesPerDetection} values each`);
    console.log(`Output shape: [${dims.join(', ')}], Calculated classes: ${numClasses}`);

    const detections: Detection[] = [];
    const boxes: number[][] = [];
    const scores: number[] = [];
    const classIds: number[] = [];

    // Parse detections
    let maxConfidenceSeen = 0;
    const allDetections = []; // Track all detections for debugging
    let debugLogged = false; // Flag to log first detection for debugging
    const isTransposed = dims[1] < dims[2];
    
    for (let i = 0; i < numDetections; i++) {
      // Access data based on format
      // Standard: data[i * valuesPerDetection + j]
      // Transposed: data[j * numDetections + i]
      const getValue = (valueIndex: number) => {
        return isTransposed 
          ? data[valueIndex * numDetections + i]
          : data[i * valuesPerDetection + valueIndex];
      };

      // Extract bbox (center x, center y, width, height)
      const x = getValue(0);
      const y = getValue(1);
      const w = getValue(2);
      const h = getValue(3);

      // YOLOv5 format: [x, y, w, h, objectness, class1, class2, ...]
      // Extract objectness score and apply sigmoid
      const objectness = this.sigmoid(getValue(4));
      
      // Extract class scores and apply sigmoid
      let maxScore = 0;
      let maxClassId = 0;
      for (let c = 0; c < numClasses; c++) {
        const classScore = this.sigmoid(getValue(5 + c));
        const finalScore = objectness * classScore; // Multiply objectness with class score
        if (finalScore > maxScore) {
          maxScore = finalScore;
          maxClassId = c;
        }
      }

      // Log first high-confidence detection for debugging
      if (!debugLogged && objectness > 0.5) {
        console.log('First high-confidence detection:', {
          objectness: objectness.toFixed(4),
          rawObjectness: getValue(4).toFixed(4),
          maxScore: maxScore.toFixed(4),
          bbox: [x, y, w, h],
          isTransposed
        });
        debugLogged = true;
      }

      // Track max confidence for debugging
      if (maxScore > maxConfidenceSeen) {
        maxConfidenceSeen = maxScore;
      }

      // Store all detections above a very low threshold for debugging
      if (maxScore > 0.01) {
        allDetections.push({
          class: this.classNames[maxClassId] || `class_${maxClassId}`,
          score: maxScore,
          classId: maxClassId
        });
      }

      // Filter by threshold
      if (maxScore > scoreThreshold) {
        boxes.push([x, y, w, h]);
        scores.push(maxScore);
        classIds.push(maxClassId);
      }
    }

    console.log(`Found ${boxes.length} detections above threshold ${scoreThreshold}`);
    console.log(`Max confidence seen: ${maxConfidenceSeen}`);
    console.log(`Total detections above 0.01 threshold: ${allDetections.length}`);

    // Show top detections for debugging
    const topDetections = allDetections
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);
    
    console.log('Top 10 detections (any confidence):', topDetections);

    // Look for lizard detections specifically
    const lizardDetections = allDetections.filter(d => 
      d.class && d.class.toLowerCase().includes('lizard')
    );
    
    if (lizardDetections.length > 0) {
      console.log('Lizard detections found:', lizardDetections);
    }

    if (boxes.length > 0) {
      console.log('Sample detection above threshold:', {
        box: boxes[0],
        boxValues: `[x=${boxes[0][0].toFixed(2)}, y=${boxes[0][1].toFixed(2)}, w=${boxes[0][2].toFixed(2)}, h=${boxes[0][3].toFixed(2)}]`,
        score: scores[0],
        class: this.classNames[classIds[0]],
        modelInputSize: this.MODEL_INPUT_SIZE,
        originalSize: `${originalWidth}x${originalHeight}`
      });
    }

    // Apply NMS
    const selectedIndices = this.nonMaxSuppression(boxes, scores, iouThreshold);
    console.log(`After NMS: ${selectedIndices.length} detections`);

    for (const idx of selectedIndices) {
      const box = boxes[idx];
      
      // Try different coordinate interpretations
      // Option 1: Model outputs are already at original image scale
      let x, y, width, height;
      
      // Check if coordinates look like they're at original scale
      const isOriginalScale = box[0] < originalWidth && box[1] < originalHeight;
      
      if (isOriginalScale) {
        console.log('Using original scale coordinates');
        // Direct use - coordinates are already at original image scale
        x = box[0] - box[2] / 2;
        y = box[1] - box[3] / 2;
        width = box[2];
        height = box[3];
      } else {
        console.log('Using 640x640 scale coordinates with letterbox conversion');
        // Original letterbox conversion
        const centerX640 = box[0];
        const centerY640 = box[1];
        const width640 = box[2];
        const height640 = box[3];
        
        // Remove letterbox padding
        const centerXNoPad = centerX640 - padX;
        const centerYNoPad = centerY640 - padY;
        
        // Scale back to original image size
        const centerX = centerXNoPad / scale;
        const centerY = centerYNoPad / scale;
        width = width640 / scale;
        height = height640 / scale;
        
        // Convert from center format to top-left corner format
        x = centerX - width / 2;
        y = centerY - height / 2;
      }

      const detection = {
        bbox: [x, y, width, height],
        class: this.classNames[classIds[idx]] || `class_${classIds[idx]}`,
        score: scores[idx],
      };

      console.log('Final detection:', {
        rawBox: `[cx=${box[0].toFixed(1)}, cy=${box[1].toFixed(1)}, w=${box[2].toFixed(1)}, h=${box[3].toFixed(1)}]`,
        coordinateMode: isOriginalScale ? 'original-scale' : '640x640-letterbox',
        originalSize: `${originalWidth}x${originalHeight}`,
        finalBbox: `[x=${x.toFixed(1)}, y=${y.toFixed(1)}, w=${width.toFixed(1)}, h=${height.toFixed(1)}]`,
        percentOfImage: `x=${(x/originalWidth*100).toFixed(1)}%, y=${(y/originalHeight*100).toFixed(1)}%, w=${(width/originalWidth*100).toFixed(1)}%, h=${(height/originalHeight*100).toFixed(1)}%`,
        class: detection.class,
        score: detection.score
      });

      detections.push(detection);
    }

    // Sort by area (larger objects first)
    detections.sort((a, b) => {
      const areaA = a.bbox[2] * a.bbox[3];
      const areaB = b.bbox[2] * b.bbox[3];
      return areaB - areaA;
    });

    return detections;
  }

  /**
   * Sigmoid activation function
   */
  private static sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  /**
   * Non-Maximum Suppression
   */
  private static nonMaxSuppression(
    boxes: number[][],
    scores: number[],
    iouThreshold: number
  ): number[] {
    // Sort by scores descending
    const indices = scores
      .map((score, idx) => ({ score, idx }))
      .sort((a, b) => b.score - a.score)
      .map(item => item.idx);

    const selected: number[] = [];
    const suppressed = new Set<number>();

    for (const idx of indices) {
      if (suppressed.has(idx)) continue;

      selected.push(idx);

      // Suppress overlapping boxes
      for (const otherIdx of indices) {
        if (otherIdx === idx || suppressed.has(otherIdx)) continue;

        const iou = this.calculateIoU(boxes[idx], boxes[otherIdx]);
        if (iou > iouThreshold) {
          suppressed.add(otherIdx);
        }
      }
    }

    return selected;
  }

  /**
   * Calculate Intersection over Union
   */
  private static calculateIoU(box1: number[], box2: number[]): number {
    // Convert from center format to corners
    const x1Min = box1[0] - box1[2] / 2;
    const y1Min = box1[1] - box1[3] / 2;
    const x1Max = box1[0] + box1[2] / 2;
    const y1Max = box1[1] + box1[3] / 2;

    const x2Min = box2[0] - box2[2] / 2;
    const y2Min = box2[1] - box2[3] / 2;
    const x2Max = box2[0] + box2[2] / 2;
    const y2Max = box2[1] + box2[3] / 2;

    // Calculate intersection
    const xMin = Math.max(x1Min, x2Min);
    const yMin = Math.max(y1Min, y2Min);
    const xMax = Math.min(x1Max, x2Max);
    const yMax = Math.min(y1Max, y2Max);

    if (xMax < xMin || yMax < yMin) return 0;

    const intersection = (xMax - xMin) * (yMax - yMin);
    const area1 = box1[2] * box1[3];
    const area2 = box2[2] * box2[3];
    const union = area1 + area2 - intersection;

    return intersection / union;
  }

  /**
   * Dispose of the model and free memory
   */
  static async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
    this.isInitialized = false;
    console.log('ONNX model disposed');
  }
}
