import type { AnnotationsData } from "../models/AnnotationsData";

export interface PreprocessedData {
  floatData: Float32Array;
  xRatio: number;
  xPad: number;
  yPad: number;
  originalWidth: number;
  originalHeight: number;
}

export class OnnxService {
  private static worker: Worker | null = null;
  private static messageIdSeq = 0;
  private static pendingResolvers = new Map<number, { resolve: (val: any) => void; reject: (err: any) => void }>();
  private static cache = new Map<string, PreprocessedData>();

  private static get MODEL_PATH() {
    return window.electronAPI?.isElectron
      ? "./models/yolo_obb_6class_h7.onnx"
      : "/models/yolo_obb_6class_h7.onnx";
  }

  private static readonly INPUT_SIZE = 1280;

  private static initializationPromise: Promise<void> | null = null;

  static async initialize(): Promise<void> {
    if (this.initializationPromise) {
      return this.initializationPromise;
    }

    this.initializationPromise = (async () => {
      console.log("Main: Spawning ONNX Web Worker...");
      this.worker = new Worker(new URL('../workers/onnx.worker.ts', import.meta.url), { type: 'module' });
      
      this.worker.onmessage = (e: MessageEvent) => {
        const { type, payload, id, error } = e.data;
        const resolver = this.pendingResolvers.get(id);
        
        if (resolver) {
          if (type === "ERROR") {
            resolver.reject(new Error(error));
          } else {
            resolver.resolve(payload);
          }
          this.pendingResolvers.delete(id);
        }
      };

      try {
        await this.postMessageAsync("INIT", { modelPath: this.MODEL_PATH });
      } catch (err) {
        this.initializationPromise = null; // Allow retry on failure
        throw err;
      }
    })();

    return this.initializationPromise;
  }

  private static postMessageAsync(type: string, payload?: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const id = ++this.messageIdSeq;
      this.pendingResolvers.set(id, { resolve, reject });
      this.worker!.postMessage({ type, payload, id });
    });
  }

  /**
   * Pre-calculates the resized and normalized float data for an image.
   * This can be called while the user is hovering or selecting files to 
   * hide the 50-200ms latency of Canvas processing.
   */
  static async preload(file: File): Promise<void> {
    const key = `${file.name}-${file.size}-${file.lastModified}`;
    if (this.cache.has(key)) return;

    try {
      const data = await this.preprocess(file);
      this.cache.set(key, data);
      console.log(`Main: Preloaded ${file.name}`);
    } catch (err) {
      console.warn(`Main: Preload failed for ${file.name}`, err);
    }
  }

  static async detect(file: File): Promise<AnnotationsData> {
    await this.initialize();
    
    const key = `${file.name}-${file.size}-${file.lastModified}`;
    let preprocessed: PreprocessedData;

    if (this.cache.has(key)) {
      console.log(`Main: Using cached preprocessed data for ${file.name}`);
      preprocessed = this.cache.get(key)!;
      this.cache.delete(key); // Remove because buffer will be transferred
    } else {
      preprocessed = await this.preprocess(file);
    }
    
    console.log("Main: Sending float data to Worker...");
    // We transfer the underlying ArrayBuffer to the worker to avoid a copy.
    return new Promise((resolve, reject) => {
      const id = ++this.messageIdSeq;
      this.pendingResolvers.set(id, { resolve, reject });
      this.worker!.postMessage(
        { type: "INFER", payload: preprocessed, id }, 
        [preprocessed.floatData.buffer] // Transferable object
      );
    });
  }

  private static async preprocess(file: File): Promise<PreprocessedData> {
    return new Promise<PreprocessedData>((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const originalWidth = img.width;
        const originalHeight = img.height;

        const scale = Math.min(this.INPUT_SIZE / originalWidth, this.INPUT_SIZE / originalHeight);
        const newW = Math.round(originalWidth * scale);
        const newH = Math.round(originalHeight * scale);

        const xPad = (this.INPUT_SIZE - newW) / 2;
        const yPad = (this.INPUT_SIZE - newH) / 2;

        const canvas = document.createElement("canvas");
        canvas.width = this.INPUT_SIZE;
        canvas.height = this.INPUT_SIZE;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (!ctx) return reject(new Error("No 2D context"));

        ctx.fillStyle = "rgb(114, 114, 114)";
        ctx.fillRect(0, 0, this.INPUT_SIZE, this.INPUT_SIZE);

        ctx.drawImage(img, xPad, yPad, newW, newH);

        const imgData = ctx.getImageData(0, 0, this.INPUT_SIZE, this.INPUT_SIZE);
        const data = imgData.data;

        const CHANNELS = 3;
        const size = this.INPUT_SIZE * this.INPUT_SIZE;
        const floatData = new Float32Array(CHANNELS * size);

        for (let i = 0; i < size; i++) {
          floatData[i] = data[i * 4] / 255.0;            // R
          floatData[size + i] = data[i * 4 + 1] / 255.0; // G
          floatData[2 * size + i] = data[i * 4 + 2] / 255.0; // B
        }

        resolve({
          floatData,
          xRatio: scale,
          xPad,
          yPad,
          originalWidth,
          originalHeight
        });
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}
