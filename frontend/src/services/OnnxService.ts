import * as ort from "onnxruntime-web";
import type { AnnotationsData } from "../models/AnnotationsData";

export class OnnxService {
  private static session: ort.InferenceSession | null = null;
  private static readonly MODEL_PATH = "/models/yolo_obb_6class_h7.onnx";
  private static readonly INPUT_SIZE = 1280;
  private static readonly CONF_THRESHOLD = 0.25;
  private static readonly IOU_THRESHOLD = 0.45;

  private static classNames = [
    "up_finger",
    "up_toe",
    "bot_finger",
    "bot_toe",
    "ruler",
    "id"
  ];

  static async initialize() {
    if (!this.session) {
      try {
        console.log("Loading ONNX Model...");
        // Use wasm backend
        ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
        this.session = await ort.InferenceSession.create(this.MODEL_PATH, {
          executionProviders: ["wasm"],
        });
        console.log("ONNX Model Loaded.");
      } catch (err) {
        console.error("Failed to load ONNX model:", err);
      }
    }
  }

  static async detect(file: File): Promise<AnnotationsData> {
    await this.initialize();
    if (!this.session) {
      throw new Error("Model not loaded");
    }

    const { tensor, xRatio, xPad, yPad, originalWidth, originalHeight } = await this.preprocess(file);

    // Pass 1: Normal inference
    console.log("Running ONNX inference (normal pass)...");
    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.session.inputNames[0]] = tensor;
    const results = await this.session.run(feeds);
    const output = results[this.session.outputNames[0]]; // Shape: [1, 11, 33600]

    // Pass 2: Flipped inference (to detect up_finger/up_toe)
    console.log("Running ONNX inference (flipped pass)...");
    const flippedTensor = this.flipTensorVertically(tensor);
    const feedsFlipped: Record<string, ort.Tensor> = {};
    feedsFlipped[this.session.inputNames[0]] = flippedTensor;
    const resultsFlipped = await this.session.run(feedsFlipped);
    const outputFlipped = resultsFlipped[this.session.outputNames[0]];
    console.log("ONNX Inference done (both passes).");

    return this.postprocessDualPass(
      output.data as Float32Array,
      outputFlipped.data as Float32Array,
      xRatio, xPad, yPad, originalWidth, originalHeight
    );
  }

  private static async preprocess(file: File) {
    return new Promise<{
      tensor: ort.Tensor;
      xRatio: number;
      xPad: number;
      yPad: number;
      originalWidth: number;
      originalHeight: number;
    }>((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const originalWidth = img.width;
        const originalHeight = img.height;

        // YOLOv8 preprocessing: Resize while preserving aspect ratio and pad to INPUT_SIZE x INPUT_SIZE
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

        // Fill with padding color (usually grey 114)
        ctx.fillStyle = "rgb(114, 114, 114)";
        ctx.fillRect(0, 0, this.INPUT_SIZE, this.INPUT_SIZE);

        ctx.drawImage(img, xPad, yPad, newW, newH);

        const imgData = ctx.getImageData(0, 0, this.INPUT_SIZE, this.INPUT_SIZE);
        const data = imgData.data;

        // Convert to Float32Array [1, 3, INPUT_SIZE, INPUT_SIZE] NCHW, normalized to [0, 1]
        const CHANNELS = 3;
        const size = this.INPUT_SIZE * this.INPUT_SIZE;
        const floatData = new Float32Array(CHANNELS * size);

        for (let i = 0; i < size; i++) {
          floatData[i] = data[i * 4] / 255.0;            // R
          floatData[size + i] = data[i * 4 + 1] / 255.0; // G
          floatData[2 * size + i] = data[i * 4 + 2] / 255.0; // B
        }

        const tensor = new ort.Tensor("float32", floatData, [1, CHANNELS, this.INPUT_SIZE, this.INPUT_SIZE]);

        resolve({
          tensor,
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

  /**
   * Flip a [1, 3, H, W] NCHW tensor vertically (reverse rows).
   */
  private static flipTensorVertically(tensor: ort.Tensor): ort.Tensor {
    const data = tensor.data as Float32Array;
    const H = this.INPUT_SIZE;
    const W = this.INPUT_SIZE;
    const flipped = new Float32Array(data.length);
    for (let c = 0; c < 3; c++) {
      const channelOffset = c * H * W;
      for (let y = 0; y < H; y++) {
        const srcRow = channelOffset + y * W;
        const dstRow = channelOffset + (H - 1 - y) * W;
        flipped.set(data.subarray(srcRow, srcRow + W), dstRow);
      }
    }
    return new ort.Tensor("float32", flipped, [1, 3, H, W]);
  }

  /**
   * Parse raw model output into detected boxes above confidence threshold.
   */
  private static parseDetections(flatData: Float32Array): {
    x: number; y: number; w: number; h: number; angle: number; conf: number; class_id: number;
  }[] {
    const num_anchors = 33600;
    const boxes: { x: number; y: number; w: number; h: number; angle: number; conf: number; class_id: number; }[] = [];

    for (let i = 0; i < num_anchors; i++) {
      let max_conf = 0.0;
      let max_class_id = -1;
      for (let c = 0; c < 6; c++) {
        const conf = flatData[(4 + c) * num_anchors + i];
        if (conf > max_conf) {
          max_conf = conf;
          max_class_id = c;
        }
      }
      if (max_conf > this.CONF_THRESHOLD) {
        boxes.push({
          x: flatData[0 * num_anchors + i],
          y: flatData[1 * num_anchors + i],
          w: flatData[2 * num_anchors + i],
          h: flatData[3 * num_anchors + i],
          angle: flatData[10 * num_anchors + i],
          conf: max_conf,
          class_id: max_class_id
        });
      }
    }
    return boxes;
  }

  /**
   * Apply NMS then keep only the best detection per class.
   */
  private static nmsAndTopOne(boxes: { x: number; y: number; w: number; h: number; angle: number; conf: number; class_id: number; }[]) {
    boxes.sort((a, b) => b.conf - a.conf);
    const afterNms: typeof boxes = [];
    for (const b of boxes) {
      let overlap = false;
      for (const k of afterNms) {
        if (k.class_id === b.class_id && this.iou(b, k) > this.IOU_THRESHOLD) {
          overlap = true;
          break;
        }
      }
      if (!overlap) afterNms.push(b);
    }
    const seen = new Set<number>();
    return afterNms.filter(b => {
      if (seen.has(b.class_id)) return false;
      seen.add(b.class_id);
      return true;
    });
  }

  private static postprocessDualPass(
    normalData: Float32Array,
    flippedData: Float32Array,
    scale: number,
    xPad: number,
    yPad: number,
    _originalWidth: number,
    originalHeight: number
  ): AnnotationsData {
    // Pass 1: Normal detections
    const normalBoxes = this.nmsAndTopOne(this.parseDetections(normalData));

    // Pass 2: Flipped detections — only keep bot_finger(2) and bot_toe(3) to use as up_finger/up_toe
    const flippedBoxes = this.nmsAndTopOne(this.parseDetections(flippedData));

    // Combine: normal pass provides bot_finger, bot_toe, ruler, id
    //          flipped pass provides up_finger (from bot_finger) and up_toe (from bot_toe)
    type DetBox = { x: number; y: number; w: number; h: number; angle: number; conf: number; class_id: number; flipped: boolean };
    const combined: DetBox[] = [];

    for (const b of normalBoxes) {
      combined.push({ ...b, flipped: false });
    }

    for (const b of flippedBoxes) {
      const name = this.classNames[b.class_id];
      // bot_finger in flipped image → up_finger in original
      // bot_toe in flipped image → up_toe in original
      if (name === "bot_finger") {
        combined.push({ ...b, class_id: 0, flipped: true }); // 0 = up_finger
      } else if (name === "bot_toe") {
        combined.push({ ...b, class_id: 1, flipped: true }); // 1 = up_toe
      }
      // Ignore ruler/id/up_* from flipped pass
    }

    // Deduplicate: keep best per class_id across both passes
    combined.sort((a, b) => b.conf - a.conf);
    const seenClasses = new Set<number>();
    const keep = combined.filter(b => {
      if (seenClasses.has(b.class_id)) return false;
      seenClasses.add(b.class_id);
      return true;
    });

    // Build output
    const output: AnnotationsData = {
      name: "",
      coords: [],
      bounding_boxes: [],
      error: undefined,
      session_id: "",
      view_type: "toepad"
    };

    keep.forEach((box, i) => {
      const corners = this.getObbCorners(box.x, box.y, box.w, box.h, box.angle);
      const class_name = this.classNames[box.class_id] || "unknown";

      let origCorners: [number, number][];
      let center_x: number;
      let center_y: number;

      if (box.flipped) {
        // Flipped pass: convert from model space → flipped original → original
        origCorners = corners.map(pt => [
          (pt[0] - xPad) / scale,
          originalHeight - 1 - (pt[1] - yPad) / scale
        ]);
        center_x = (box.x - xPad) / scale;
        center_y = originalHeight - 1 - (box.y - yPad) / scale;
      } else {
        origCorners = corners.map(pt => [
          (pt[0] - xPad) / scale,
          (pt[1] - yPad) / scale
        ]);
        center_x = (box.x - xPad) / scale;
        center_y = (box.y - yPad) / scale;
      }

      // Ruler: 2 endpoint landmarks
      if (class_name === "ruler" || class_name === "scale") {
        const dist01 = Math.hypot(origCorners[0][0] - origCorners[1][0], origCorners[0][1] - origCorners[1][1]);
        const dist12 = Math.hypot(origCorners[1][0] - origCorners[2][0], origCorners[1][1] - origCorners[2][1]);

        let mid1: [number, number];
        let mid2: [number, number];

        if (dist01 < dist12) {
          mid1 = [(origCorners[0][0] + origCorners[1][0]) / 2.0, (origCorners[0][1] + origCorners[1][1]) / 2.0];
          mid2 = [(origCorners[2][0] + origCorners[3][0]) / 2.0, (origCorners[2][1] + origCorners[3][1]) / 2.0];
        } else {
          mid1 = [(origCorners[1][0] + origCorners[2][0]) / 2.0, (origCorners[1][1] + origCorners[2][1]) / 2.0];
          mid2 = [(origCorners[3][0] + origCorners[0][0]) / 2.0, (origCorners[3][1] + origCorners[0][1]) / 2.0];
        }

        if (mid1[0] > mid2[0]) {
          [mid1, mid2] = [mid2, mid1];
        }

        output.coords.push({ x: mid1[0], y: mid1[1], id: 17 });
        output.coords.push({ x: mid2[0], y: mid2[1], id: 18 });
      } else if (class_name !== "id") {
        output.coords.push({ x: center_x, y: center_y, id: i });
      }

      output.bounding_boxes!.push({
        id: i,
        left: Math.min(...origCorners.map(p => p[0])),
        top: Math.min(...origCorners.map(p => p[1])),
        width: Math.max(...origCorners.map(p => p[0])) - Math.min(...origCorners.map(p => p[0])),
        height: Math.max(...origCorners.map(p => p[1])) - Math.min(...origCorners.map(p => p[1])),
        label: class_name,
        confidence: box.conf,
        obb_corners: origCorners.map(p => ({ x: p[0], y: p[1] }))
      });
    });

    console.log(`Detected ${keep.length} classes: ${keep.map(b => this.classNames[b.class_id]).join(', ')}`);
    return output;
  }

  // Simplified IoU calculation based on horizontal bounds ONLY, works as a quick AABB NMS
  private static iou(a: {x: number; y: number; w: number; h: number}, b: {x: number; y: number; w: number; h: number}) {
    // Treat as AABB covering max width/height
    const aMinX = a.x - a.w / 2;
    const aMaxX = a.x + a.w / 2;
    const aMinY = a.y - a.h / 2;
    const aMaxY = a.y + a.h / 2;
    
    const bMinX = b.x - b.w / 2;
    const bMaxX = b.x + b.w / 2;
    const bMinY = b.y - b.h / 2;
    const bMaxY = b.y + b.h / 2;

    const interX1 = Math.max(aMinX, bMinX);
    const interY1 = Math.max(aMinY, bMinY);
    const interX2 = Math.min(aMaxX, bMaxX);
    const interY2 = Math.min(aMaxY, bMaxY);

    if (interX2 <= interX1 || interY2 <= interY1) return 0;

    const interArea = (interX2 - interX1) * (interY2 - interY1);
    const aArea = a.w * a.h;
    const bArea = b.w * b.h;

    return interArea / (aArea + bArea - interArea);
  }

  // Gets the 4 corners of rotated rectangle
  private static getObbCorners(cx: number, cy: number, w: number, h: number, angle: number): [number, number][] {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);

    const corners: [number, number][] = [
      [-w / 2, -h / 2],
      [w / 2, -h / 2],
      [w / 2, h / 2],
      [-w / 2, h / 2]
    ];

    return corners.map(([x, y]) => [
      cx + x * cos - y * sin,
      cy + x * sin + y * cos
    ]);
  }
}
