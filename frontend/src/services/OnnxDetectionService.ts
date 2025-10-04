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
  private static readonly DEFAULT_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
  ];

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

      const defaultModelUrl = modelUrl || '/models/yolov5n-seg.onnx';
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
   * Preprocess image to model input format
   */
  private static preprocessImage(imageElement: HTMLImageElement | HTMLCanvasElement): {
    tensor: ort.Tensor;
    originalWidth: number;
    originalHeight: number;
  } {
    // Create canvas for image processing
    const canvas = document.createElement('canvas');
    canvas.width = this.MODEL_INPUT_SIZE;
    canvas.height = this.MODEL_INPUT_SIZE;
    const ctx = canvas.getContext('2d')!;

    // Draw and resize image
    const originalWidth = imageElement.width || (imageElement as HTMLImageElement).naturalWidth;
    const originalHeight = imageElement.height || (imageElement as HTMLImageElement).naturalHeight;

    ctx.drawImage(imageElement, 0, 0, this.MODEL_INPUT_SIZE, this.MODEL_INPUT_SIZE);

    // Get image data
    const imageData = ctx.getImageData(0, 0, this.MODEL_INPUT_SIZE, this.MODEL_INPUT_SIZE);
    const pixels = imageData.data;

    // Convert to NCHW format and normalize to [0, 1]
    const red: number[] = [];
    const green: number[] = [];
    const blue: number[] = [];

    for (let i = 0; i < pixels.length; i += 4) {
      red.push(pixels[i] / 255.0);
      green.push(pixels[i + 1] / 255.0);
      blue.push(pixels[i + 2] / 255.0);
    }

    const input = Float32Array.from([...red, ...green, ...blue]);
    const tensor = new ort.Tensor('float32', input, [1, 3, this.MODEL_INPUT_SIZE, this.MODEL_INPUT_SIZE]);

    return { tensor, originalWidth, originalHeight };
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
      const { tensor, originalWidth, originalHeight } = this.preprocessImage(imageElement);

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
    scoreThreshold: number,
    iouThreshold: number
  ): Detection[] {
    const data = output.data as Float32Array;
    const dims = output.dims;

    // YOLOv5 output format: [batch, num_detections, values_per_detection]
    // For YOLOv5n-seg: [1, 25200, 117] where 117 = 4 bbox + 1 objectness + 80 classes + 32 mask
    const numDetections = dims[1];
    const valuesPerDetection = dims[2];
    const numClasses = 80;

    console.log(`Processing ${numDetections} detections with ${valuesPerDetection} values each`);

    const detections: Detection[] = [];
    const boxes: number[][] = [];
    const scores: number[] = [];
    const classIds: number[] = [];

    // Parse detections - format is [1, num_detections, 117]
    let maxConfidenceSeen = 0;
    const allDetections = []; // Track all detections for debugging
    
    for (let i = 0; i < numDetections; i++) {
      const offset = i * valuesPerDetection;

      // Extract bbox (center x, center y, width, height)
      const x = data[offset];
      const y = data[offset + 1];
      const w = data[offset + 2];
      const h = data[offset + 3];

      // Extract class scores (positions 4-83, no separate objectness for this model)
      let maxScore = 0;
      let maxClassId = 0;
      for (let c = 0; c < numClasses; c++) {
        const classScore = data[offset + 4 + c];
        if (classScore > maxScore) {
          maxScore = classScore;
          maxClassId = c;
        }
      }

      // Track max confidence for debugging
      if (maxScore > maxConfidenceSeen) {
        maxConfidenceSeen = maxScore;
      }

      // Store all detections above a very low threshold for debugging
      if (maxScore > 0.01) {
        allDetections.push({
          class: this.classNames[maxClassId],
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

    // Look for flower-related classes specifically
    const flowerRelated = allDetections.filter(d => 
      d.class.toLowerCase().includes('plant') || 
      d.class.toLowerCase().includes('flower') ||
      d.class.toLowerCase().includes('vase') ||
      d.class.toLowerCase().includes('potted')
    );
    
    if (flowerRelated.length > 0) {
      console.log('Flower-related detections:', flowerRelated);
    }

    if (boxes.length > 0) {
      console.log('Sample detection above threshold:', {
        box: boxes[0],
        score: scores[0],
        class: this.classNames[classIds[0]]
      });
    }

    // Apply NMS
    const selectedIndices = this.nonMaxSuppression(boxes, scores, iouThreshold);
    console.log(`After NMS: ${selectedIndices.length} detections`);

    for (const idx of selectedIndices) {
      const box = boxes[idx];
      // Convert from center format (x, y, w, h) to corner format and scale to original size
      const scaleX = originalWidth / this.MODEL_INPUT_SIZE;
      const scaleY = originalHeight / this.MODEL_INPUT_SIZE;

      const x = (box[0] - box[2] / 2) * scaleX;
      const y = (box[1] - box[3] / 2) * scaleY;
      const width = box[2] * scaleX;
      const height = box[3] * scaleY;

      detections.push({
        bbox: [x, y, width, height],
        class: this.classNames[classIds[idx]] || `class_${classIds[idx]}`,
        score: scores[idx],
      });
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
