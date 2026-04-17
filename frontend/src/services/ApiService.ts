
import type { AnnotationsData } from "../models/AnnotationsData";
import type { ImageSet } from "../models/ImageSet";
import { SessionService } from "./SessionService";
import { API_URL, getApiUrl } from "./config";

async function apiUrl(): Promise<string> {
  if (window.electronAPI?.isElectron) {
    return getApiUrl();
  }
  return API_URL;
}

export class ApiService {
  /**
   * Initialize session before making API calls
   */
  static async initialize(): Promise<void> {
    await SessionService.initializeSession();
  }

  static async uploadMultipleImages(
    files: File[], 
    viewType: string, 
    toepadPredictorType?: string
  ): Promise<AnnotationsData[]> {
    const clientAnnotations: AnnotationsData[] = [];

    const formData = new FormData();
    files.forEach((file) => {
      formData.append("image", file);
    });
    formData.append("view_type", viewType === "toepads" ? "toepad" : viewType);
    if (viewType === "free") {
      formData.append("skip_prediction", "true");
    }
    // Add toepad predictor type if specified
    if (viewType === "toepads" && toepadPredictorType) {
      formData.append("toepad_predictor_type", toepadPredictorType);
    }
    if (clientAnnotations.length > 0) {
      formData.append("client_annotations", JSON.stringify(clientAnnotations));
    }
    const base = await apiUrl();
    const res = await fetch(`${base}/data`, {
      method: "POST",
      headers: {
        ...SessionService.getSessionHeaders(),
      },
      body: formData,
    });
    if (!res.ok) {
      const errorResult = await res.json();
      throw new Error(errorResult.error ?? "Failed to process images");
    }

    // If the server returns valid data, we return it. 
    // However, if we passed client_annotations, the server will just echo them back along with processed images
    // which is perfectly fine.
    return res.json() as Promise<AnnotationsData[]>;
  }
  static async fetchImageSet(imageFilename: string): Promise<ImageSet> {
    const base = await apiUrl();
    // Validate session
    const sessionId = SessionService.getSessionId();
    if (!sessionId) {
      throw new Error("No active session");
    }

    // Instead of downloading base64 images via /image, we directly point to the new /image_file endpoint
    // This returns an HTTP URL which natively evades Electron's strict file:// + data URI restrictions inside SVGs
    // By passing X-Session-ID, we might need a query param to guarantee cross-origin retrieval
    const buildUrl = (type: string) => {
      // Append a timestamp to prevent aggressive browser caching
      return `${base}/image_file?image_filename=${encodeURIComponent(
        imageFilename
      )}&type=${type}&session_id=${sessionId}&_t=${Date.now()}`;
    };

    return {
      original: buildUrl("original"),
      inverted: buildUrl("inverted"),
      color_contrasted: buildUrl("color_contrasted"),
    };
  }
  static async fetchUploadedFiles(): Promise<{ filename: string; view_type: string }[]> {
    const base = await apiUrl();
    const res = await fetch(`${base}/list_uploads`, {
      method: "GET",
      headers: {
        ...SessionService.getSessionHeaders(),
      },
    });
    if (!res.ok) throw new Error("Failed to fetch uploaded files");
    return res.json();
  }

  static async processExistingImage(
    filename: string,
    viewType: string,
    toepadPredictorType?: string
  ): Promise<AnnotationsData> {
    const base = await apiUrl();
    const viewTypeParam = viewType === "toepads" ? "toepad" : viewType;
    let url = `${base}/process_existing?filename=${encodeURIComponent(filename)}&view_type=${encodeURIComponent(viewTypeParam)}`;
    if (viewType === "toepads" && toepadPredictorType) {
      url += `&toepad_predictor_type=${encodeURIComponent(toepadPredictorType)}`;
    }
    const res = await fetch(url, {
      method: "POST",
      headers: {
        ...SessionService.getSessionHeaders(),
      },
    });
    if (!res.ok) throw new Error("Failed to process existing image");
    return res.json();
  }

  static async saveAnnotations(
    payload: AnnotationsData
  ): Promise<{ success: boolean }> {
    const base = await apiUrl();
    const res = await fetch(`${base}/save_annotations`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...SessionService.getSessionHeaders(),
      },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Failed to save annotations");
    return res.json();
  }

  static async downloadAnnotatedImage(imageUrl: string): Promise<Blob> {
    const res = await fetch(imageUrl);
    if (!res.ok) throw new Error("Failed to fetch annotated image");
    return res.blob();
  }
  static async exportScatterData(payload: {
    coords: { x: number; y: number }[];
    name: string;
  }): Promise<{ image_urls?: string[] }> {
    const base = await apiUrl();
    const res = await fetch(`${base}/endpoint`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...SessionService.getSessionHeaders(),
      },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Failed to export scatter data");
    return res.json();
  }

  static async clearHistory(): Promise<{ success: boolean }> {
    const base = await apiUrl();
    const res = await fetch(`${base}/clear_history`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...SessionService.getSessionHeaders(),
      },
    });
    if (!res.ok) throw new Error("Failed to clear history");
    return res.json();
  }

  /**
   * Get current session information
   */
  static async getSessionInfo(): Promise<SessionInfo> {
    return await SessionService.getSessionInfo();
  }

  /**
   * Extract ID from an image using OCR on an ID bounding box
   */
  static async extractId(imageFilename: string, idBox?: { left: number; top: number; width: number; height: number }): Promise<ExtractIdResult> {
    const formData = new URLSearchParams();
    formData.append("image_filename", imageFilename);
    if (idBox) {
      formData.append("id_box", JSON.stringify(idBox));
    }

    const base = await apiUrl();
    const res = await fetch(`${base}/extract_id`, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        ...SessionService.getSessionHeaders(),
      },
      body: formData,
    });

    const bodyText = await res.text();
    if (!res.ok) {
      let message = `extract_id failed (${res.status})`;
      try {
        const errJson = JSON.parse(bodyText) as { error?: string };
        if (errJson?.error) message = errJson.error;
      } catch {
        if (bodyText.trim() && !bodyText.trim().startsWith("<")) {
          message = bodyText.trim().slice(0, 200);
        }
      }
      throw new Error(message);
    }
    try {
      return JSON.parse(bodyText) as ExtractIdResult;
    } catch {
      throw new Error("Invalid JSON from extract_id");
    }
  }

  static async listPredictors(): Promise<PredictorMeta[]> {
    const base = await apiUrl();
    const res = await fetch(`${base}/predictors`, {
      method: "GET",
      headers: {
        ...SessionService.getSessionHeaders(),
      },
    });
    if (!res.ok) throw new Error("Failed to list predictors");
    const data = (await res.json()) as {
      success: boolean;
      predictors: PredictorMeta[];
      error?: string;
    };
    if (!data.success) throw new Error(data.error ?? "Failed to list predictors");
    return data.predictors ?? [];
  }

  static async uploadPredictor(file: File): Promise<PredictorMeta> {
    const formData = new FormData();
    formData.append("predictor", file);
    const base = await apiUrl();
    const res = await fetch(`${base}/predictors`, {
      method: "POST",
      headers: {
        ...SessionService.getSessionHeaders(),
      },
      body: formData,
    });
    const data = (await res.json()) as {
      success: boolean;
      predictor?: PredictorMeta;
      error?: string;
    };
    if (!res.ok || !data.success || !data.predictor) {
      throw new Error(data.error ?? "Failed to upload predictor");
    }
    return data.predictor;
  }

  static async deletePredictor(id: string): Promise<void> {
    const base = await apiUrl();
    const res = await fetch(`${base}/predictors/${encodeURIComponent(id)}`, {
      method: "DELETE",
      headers: {
        ...SessionService.getSessionHeaders(),
      },
    });
    const data = (await res.json()) as { success?: boolean; error?: string };
    if (!res.ok || data.success !== true) {
      throw new Error(data.error ?? "Failed to delete predictor");
    }
  }

  static async freeAutoplace(
    filename: string,
    predictorId: string
  ): Promise<AnnotationsData> {
    const base = await apiUrl();
    const url = `${base}/free_autoplace?filename=${encodeURIComponent(filename)}`;
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...SessionService.getSessionHeaders(),
      },
      body: JSON.stringify({ predictor_id: predictorId }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error ?? "Failed to auto-place landmarks");
    }
    return res.json();
  }
}

interface ExtractIdResult {
  success: boolean;
  id?: string;
  confidence?: number;
  error?: string;
}

interface SessionInfo {
  success: boolean;
  session_id: string;
  session_id_short: string;
  created_at: string;
  session_folder: string;
  file_count: number;
}

export interface PredictorMeta {
  id: string;
  display_name: string;
  stored_filename?: string;
  uploaded_at?: string;
  size_bytes?: number;
  num_parts?: number | null;
}
