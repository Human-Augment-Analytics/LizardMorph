import type { ImageSetResponse } from "../models/ImageSetResponse";
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
    let clientAnnotations: AnnotationsData[] = [];
    if ((viewType === "toepads" || viewType === "toepad") && !window.electronAPI?.isElectron) {
      try {
        const { OnnxService } = await import("./OnnxService");
        for (const file of files) {
          const ann = await OnnxService.detect(file);
          ann.name = file.name;
          clientAnnotations.push(ann);
        }
      } catch (err) {
        console.error("Local ONNX inference failed, will fallback to server inference:", err);
      }
    }

    const formData = new FormData();
    files.forEach((file) => {
      formData.append("image", file);
    });
    formData.append("view_type", viewType === "toepads" ? "toepad" : viewType);
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
    const res = await fetch(
      `${base}/image?image_filename=${encodeURIComponent(imageFilename)}`,
      {
        method: "POST",
        headers: {
          ...SessionService.getSessionHeaders(),
        },
      }
    );
    if (!res.ok) throw new Error("Image set fetch failed");

    const result: ImageSetResponse & { error?: string } = await res.json();
    if (result.error) {
      throw new Error(result.error);
    }

    const fileExtension = imageFilename.split(".").pop()?.toLowerCase() ?? "";
    let mimeType: string;
    if (fileExtension === "png") {
      mimeType = "image/png";
    } else if (fileExtension === "gif") {
      mimeType = "image/gif";
    } else {
      mimeType = "image/jpeg";
    }

    return {
      original: `data:${mimeType};base64,${result.image3}`,
      inverted: `data:${mimeType};base64,${result.image2}`,
      color_contrasted: `data:${mimeType};base64,${result.image1}`,
    };
  }
  static async fetchUploadedFiles(): Promise<string[]> {
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
   * Extract ID from an image using YOLO detection and OCR
   */
  static async extractId(imageFilename: string): Promise<ExtractIdResult> {
    const formData = new URLSearchParams();
    formData.append("image_filename", imageFilename);

    const base = await apiUrl();
    const res = await fetch(`${base}/extract_id`, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        ...SessionService.getSessionHeaders(),
      },
      body: formData,
    });

    if (!res.ok) {
      const errorResult = await res.json();
      throw new Error(errorResult.error ?? "Failed to extract ID");
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
