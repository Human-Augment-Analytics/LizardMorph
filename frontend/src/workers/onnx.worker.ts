import * as ort from 'onnxruntime-web';
import type { AnnotationsData } from "../models/AnnotationsData";

const numThreads = Math.min(8, navigator.hardwareConcurrency || 4);
console.log(`Worker: Setting WASM threads to ${numThreads}. SharedArrayBuffer available: ${typeof SharedArrayBuffer !== 'undefined'}`);
ort.env.wasm.numThreads = numThreads;
ort.env.wasm.proxy = false; // Prevents recursive worker creation in some environments

// We define them locally inside worker to not import DOM types if not needed
const INPUT_SIZE = 1280;
const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;

const classNames = [
  "up_finger",
  "up_toe",
  "bot_finger",
  "bot_toe",
  "ruler",
  "id"
];

let session: ort.InferenceSession | null = null;

async function initialize(modelPath: string) {
  if (!session) {
    console.log("Worker: Loading ONNX Model...", modelPath);
    // Explicitly set WASM paths if provided (critical for Electron file:// URLs)
    // In this project, the .onnx files are in public/models/ but .wasm files are in public/
    let basePath = modelPath.substring(0, modelPath.lastIndexOf("/") + 1);
    
    // If we're in a subdirectory, check if wasm files are at the root
    if (basePath.endsWith("/models/")) {
      basePath = basePath.slice(0, -7); // Remove "models/"
    } else if (basePath.endsWith("models/")) {
      basePath = basePath.slice(0, -7); 
    }
    
    console.log("Worker: Setting WASM paths to:", basePath);
    ort.env.wasm.wasmPaths = basePath;
    
    session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ["wasm"],
    });
    console.log("Worker: ONNX Model Loaded.");
  }
}

function flipTensorVertically(data: Float32Array): Float32Array {
  const H = INPUT_SIZE;
  const W = INPUT_SIZE;
  const HW = H * W;
  const flipped = new Float32Array(data.length);
  
  for (let c = 0; c < 3; c++) {
    const channelOffset = c * HW;
    for (let y = 0; y < H; y++) {
      const srcRowStart = channelOffset + y * W;
      const dstRowStart = channelOffset + (H - 1 - y) * W;
      flipped.set(data.subarray(srcRowStart, srcRowStart + W), dstRowStart);
    }
  }
  return flipped;
}

function parseDetections(flatData: Float32Array): {
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
    if (max_conf > CONF_THRESHOLD) {
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

function iou(a: {x: number; y: number; w: number; h: number}, b: {x: number; y: number; w: number; h: number}) {
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

function nmsAndTopOne(boxes: { x: number; y: number; w: number; h: number; angle: number; conf: number; class_id: number; }[]) {
  boxes.sort((a, b) => b.conf - a.conf);
  const afterNms: typeof boxes = [];
  for (const b of boxes) {
    let overlap = false;
    for (const k of afterNms) {
      if (k.class_id === b.class_id && iou(b, k) > IOU_THRESHOLD) {
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

function getObbCorners(cx: number, cy: number, w: number, h: number, angle: number): [number, number][] {
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

function postprocessDualPass(
  normalData: Float32Array,
  flippedData: Float32Array,
  scale: number,
  xPad: number,
  yPad: number,
  originalHeight: number
): AnnotationsData {
  const normalBoxes = nmsAndTopOne(parseDetections(normalData));
  const flippedBoxes = nmsAndTopOne(parseDetections(flippedData));

  type DetBox = { x: number; y: number; w: number; h: number; angle: number; conf: number; class_id: number; flipped: boolean };
  const combined: DetBox[] = [];

  for (const b of normalBoxes) {
    combined.push({ ...b, flipped: false });
  }

  for (const b of flippedBoxes) {
    const name = classNames[b.class_id];
    if (name === "bot_finger") {
      combined.push({ ...b, class_id: 0, flipped: true });
    } else if (name === "bot_toe") {
      combined.push({ ...b, class_id: 1, flipped: true });
    }
  }

  combined.sort((a, b) => b.conf - a.conf);
  const seenClasses = new Set<number>();
  const keep = combined.filter(b => {
    if (seenClasses.has(b.class_id)) return false;
    seenClasses.add(b.class_id);
    return true;
  });

  const output: AnnotationsData = {
    name: "",
    coords: [],
    bounding_boxes: [],
    error: undefined,
    session_id: "",
    view_type: "toepad"
  };

  keep.forEach((box, i) => {
    const corners = getObbCorners(box.x, box.y, box.w, box.h, box.angle);
    const class_name = classNames[box.class_id] || "unknown";

    let origCorners: [number, number][];
    let center_x: number;
    let center_y: number;

    if (box.flipped) {
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

      const dx = mid2[0] - mid1[0];
      const dy = mid2[1] - mid1[1];
      const length = Math.hypot(dx, dy);
      const dirX = length > 0 ? dx / length : 1.0;
      const dirY = length > 0 ? dy / length : 0.0;

      const scaleBarLengthMm = 10.0;
      // Physical ruler is 1.55mm longer than the marked scale bar (0.55mm left, 1mm right)
      const totalLengthMm = scaleBarLengthMm + 1.55;
      const pixelsPerMm = length / totalLengthMm;
      const leftOffsetPixels = pixelsPerMm * 0.55;
      const rightOffsetPixels = pixelsPerMm * 1.0;

      // pt1 is at 0mm (0.55mm offset on left)
      // pt2 is at 10mm (1.0mm offset from right edge)
      const pt1 = [mid1[0] + dirX * leftOffsetPixels, mid1[1] + dirY * leftOffsetPixels];
      const pt2 = [mid2[0] - dirX * rightOffsetPixels, mid2[1] - dirY * rightOffsetPixels];

      output.coords.push({ x: pt1[0], y: pt1[1], id: 0 });
      output.coords.push({ x: pt2[0], y: pt2[1], id: 1 });
    } else if (class_name !== "id") {
      // For fingers/toes, we use a different ID range or just ensure it doesn't collide
      // Actually, to respect "Landmark 1 & 2" and have unique IDs, we should offset others
      output.coords.push({ x: center_x, y: center_y, id: box.class_id + 10 });
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

  console.log(`Worker: Detected ${keep.length} classes: ${keep.map(b => classNames[b.class_id]).join(', ')}`);
  return output;
}

self.onmessage = async (e: MessageEvent) => {
  const { type, payload, id } = e.data;
  
  try {
    if (type === "INIT") {
      await initialize(payload.modelPath);
      self.postMessage({ type: "INIT_DONE", id });
    } else if (type === "INFER") {
      if (!session) throw new Error("Worker: Model not initialized");

      const { floatData, xRatio, xPad, yPad, originalHeight } = payload;
      
      const normalTensor = new ort.Tensor("float32", floatData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
      const flippedFloatData = flipTensorVertically(floatData);
      const flippedTensor = new ort.Tensor("float32", flippedFloatData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
      const inputName = session.inputNames[0];

      console.log("Worker: Running sequential passes (Normal + Flipped)...");
      const startTime = performance.now();
      
      const results = await session.run({ [inputName]: normalTensor });
      const resultsFlipped = await session.run({ [inputName]: flippedTensor });
      
      const endTime = performance.now();
      console.log(`Worker: Sequential ONNX Inference done in ${(endTime - startTime).toFixed(2)}ms.`);

      const outputName = session.outputNames[0];
      const normalOutput = results[outputName].data as Float32Array;
      const flippedOutput = resultsFlipped[outputName].data as Float32Array;
      
      const annotations = postprocessDualPass(
        normalOutput,
        flippedOutput,
        xRatio, xPad, yPad, originalHeight
      );

      self.postMessage({ type: "RESULT", payload: annotations, id });
    }
  } catch (error: any) {
    self.postMessage({ type: "ERROR", error: error.message || error.toString(), id });
  }
};
