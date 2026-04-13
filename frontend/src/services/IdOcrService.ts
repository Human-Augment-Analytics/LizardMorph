import Tesseract from "tesseract.js";
import { SessionService } from "./SessionService";

export type IdBoxPixels = { left: number; top: number; width: number; height: number };

export interface ClientExtractIdResult {
  success: boolean;
  id?: string;
  confidence?: number;
  error?: string;
}

let digitsWorkerPromise: Promise<Tesseract.Worker> | null = null;

async function getDigitsWorker(): Promise<Tesseract.Worker> {
  if (!digitsWorkerPromise) {
    digitsWorkerPromise = (async () => {
      const w = await Tesseract.createWorker("eng");
      await w.setParameters({
        tessedit_char_whitelist: "0123456789",
        tessedit_pageseg_mode: Tesseract.PSM.SINGLE_LINE,
      });
      return w;
    })();
  }
  return digitsWorkerPromise;
}

/** Sliding 3–4 digit candidates in 0..2000 (aligned with backend id_extractor). */
function bestIdFromDigitString(digits: string, conf: number): { id: string; confidence: number } | null {
  if (digits.length < 3) return null;
  let best: { id: string; confidence: number } | null = null;
  const windows =
    digits.length > 4
      ? Array.from({ length: digits.length - 3 }, (_, i) => digits.slice(i, i + 4))
      : [digits];
  for (const c of windows) {
    if (c.length < 3 || c.length > 4 || !/^\d+$/.test(c)) continue;
    const n = parseInt(c, 10);
    if (n < 0 || n > 2000) continue;
    if (!best || c.length > best.id.length) {
      best = { id: c, confidence: conf };
    }
  }
  return best;
}

function parseRecognizeResult(text: string, confidence01: number): { id: string; confidence: number } | null {
  const digits = text.replace(/\D/g, "");
  return bestIdFromDigitString(digits, confidence01);
}

async function loadImageBitmapFromUrl(url: string): Promise<ImageBitmap> {
  const res = await fetch(url, {
    credentials: "include",
    headers: SessionService.getSessionHeaders(),
  });
  if (!res.ok) throw new Error(`Failed to load image for OCR (${res.status})`);
  const blob = await res.blob();
  return createImageBitmap(blob);
}

function cropToCanvas(bmp: ImageBitmap, box: IdBoxPixels): HTMLCanvasElement {
  const pad = 0.02 * Math.max(box.width, box.height, 1);
  const x = Math.max(0, Math.floor(box.left - pad));
  const y = Math.max(0, Math.floor(box.top - pad));
  const w = Math.min(bmp.width - x, Math.ceil(box.width + 2 * pad));
  const h = Math.min(bmp.height - y, Math.ceil(box.height + 2 * pad));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, w);
  canvas.height = Math.max(1, h);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas not available");
  ctx.drawImage(bmp, x, y, w, h, 0, 0, canvas.width, canvas.height);
  return canvas;
}

async function recognizeCanvas(canvas: HTMLCanvasElement): Promise<{ text: string; confidence: number }> {
  const worker = await getDigitsWorker();
  const {
    data: { text, confidence },
  } = await worker.recognize(canvas);
  return { text, confidence: confidence / 100 };
}

/**
 * Read toe-pad ID digits from the session image URL and a pixel bounding box (client YOLO).
 * Runs in the browser (tesseract.js) — no server OCR required when the ID box is known.
 */
export async function extractIdFromImageUrl(
  imageUrl: string,
  box: IdBoxPixels
): Promise<ClientExtractIdResult> {
  let bmp: ImageBitmap | null = null;
  try {
    bmp = await loadImageBitmapFromUrl(imageUrl);
    const canvas = cropToCanvas(bmp, box);

    let { text, confidence } = await recognizeCanvas(canvas);
    let parsed = parseRecognizeResult(text, confidence);
    const threshold = 0.5;
    if (!parsed || confidence < threshold) {
      const flipped = document.createElement("canvas");
      flipped.width = canvas.width;
      flipped.height = canvas.height;
      const fctx = flipped.getContext("2d");
      if (fctx) {
        fctx.translate(flipped.width, flipped.height);
        fctx.rotate(Math.PI);
        fctx.drawImage(canvas, 0, 0);
        const r2 = await recognizeCanvas(flipped);
        const p2 = parseRecognizeResult(r2.text, r2.confidence);
        if (p2 && r2.confidence >= confidence) {
          text = r2.text;
          confidence = r2.confidence;
          parsed = p2;
        }
      }
    }

    if (parsed?.id) {
      return {
        success: true,
        id: parsed.id,
        confidence: parsed.confidence,
      };
    }
    return {
      success: false,
      error: "No ID found in the provided bounding box",
    };
  } catch (e) {
    return {
      success: false,
      error: e instanceof Error ? e.message : "Client OCR failed",
    };
  } finally {
    bmp?.close();
  }
}
