import type { ImageSetResponse } from "../models/ImageSetResponse";
import type { AnnotationsData } from "../models/AnnotationsData";
import type { ImageSet } from "../models/ImageSet";
import { SessionService } from "./SessionService";
import { API_URL } from "./config";

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
    const formData = new FormData();
    files.forEach((file) => {
      formData.append("image", file);
    });
    formData.append("view_type", viewType === "toepads" ? "toepad" : viewType);
    // Add toepad predictor type if specified
    if (viewType === "toepads" && toepadPredictorType) {
      formData.append("toepad_predictor_type", toepadPredictorType);
    }
    const res = await fetch(`${API_URL}/data`, {
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
    return res.json() as Promise<AnnotationsData[]>;
  }
  static async fetchImageSet(imageFilename: string): Promise<ImageSet> {
    const res = await fetch(
      `${API_URL}/image?image_filename=${encodeURIComponent(imageFilename)}`,
      {
        method: "POST",
        headers: {
          "Access-Control-Allow-Origin": "*",
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
    const res = await fetch(`${API_URL}/list_uploads`, {
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
    const viewTypeParam = viewType === "toepads" ? "toepad" : viewType;
    let url = `${API_URL}/process_existing?filename=${encodeURIComponent(filename)}&view_type=${encodeURIComponent(viewTypeParam)}`;
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
    const res = await fetch(`${API_URL}/save_annotations`, {
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
    const res = await fetch(`${API_URL}/endpoint`, {
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
    const res = await fetch(`${API_URL}/clear_history`, {
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
}

interface SessionInfo {
  success: boolean;
  session_id: string;
  session_id_short: string;
  created_at: string;
  session_folder: string;
  file_count: number;
}
