
import type { AnnotationsData } from "../models/AnnotationsData";
import type { ImageSet } from "../models/ImageSet";
import { SessionService } from "./SessionService";
import { getApiUrl } from "./config";

async function apiUrl(): Promise<string> {
  return getApiUrl();
}

function buildEndpointUrl(base: string, endpoint: string): string {
  const cleanBase = base.replace(/\/+$/, "");
  const cleanEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  if (cleanBase.endsWith("/api") && cleanEndpoint.startsWith("/api/")) {
    return `${cleanBase}${cleanEndpoint.slice(4)}`;
  }
  return `${cleanBase}${cleanEndpoint}`;
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
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || "Failed to fetch uploaded files");
    }
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
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `Failed to process existing image (${res.status} ${res.statusText})`);
    }
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
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || "Failed to save annotations");
    }
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
    const url = buildEndpointUrl(base, "/api/predictors");
    const res = await fetch(url, {
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
    const url = buildEndpointUrl(base, "/api/predictors");
    const res = await fetch(url, {
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
    const url = buildEndpointUrl(base, `/api/predictors/${encodeURIComponent(id)}`);
    const res = await fetch(url, {
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
    const url = buildEndpointUrl(base, `/api/free_autoplace?filename=${encodeURIComponent(filename)}`);
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

  static async trainPredictor(
    modelName: string,
    file: File,
    options?: {
      nu?: number;
      tree_depth?: number;
      cascade_depth?: number;
      oversampling_amount?: number;
      feature_pool_size?: number;
      num_test_splits?: number;
      test_split?: number;
    }
  ): Promise<{ success: boolean; job_id: string; message: string }> {
    const base = await apiUrl();
    const formData = new FormData();
    formData.append("model_name", modelName);
    formData.append("dataset", file);

    if (options) {
      Object.entries(options).forEach(([key, val]) => {
        if (val !== undefined && val !== null) {
          formData.append(key, String(val));
        }
      });
    }

    const response = await fetch(`${base}/train_predictor`, {
      method: "POST",
      headers: {
        ...SessionService.getSessionHeaders(),
      },
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: "Server error" }));
      throw new Error(err.error || "Failed to start training job");
    }

    return response.json();
  }

  static async getTrainStatus(
    jobId: string
  ): Promise<{
    success: boolean;
    status: "pending" | "training" | "completed" | "failed";
    error: string | null;
    predictor: PredictorMeta | null;
  }> {
    const base = await apiUrl();
    const response = await fetch(`${base}/train_status/${encodeURIComponent(jobId)}`, {
      headers: {
        ...SessionService.getSessionHeaders(),
      },
    });
    if (!response.ok) {
      throw new Error("Failed to get training status");
    }
    return response.json();
  }

  static async getModels(): Promise<ModelVersionItem[]> {
    const base = await apiUrl();
    const url = buildEndpointUrl(base, "/api/models");
    const response = await fetch(url, {
      headers: {
        ...SessionService.getSessionHeaders(),
      },
    });
    if (!response.ok) {
      throw new Error("Failed to fetch models");
    }
    const data = await response.json();
    return data.models || [];
  }

  static async deleteModel(modelId: string): Promise<void> {
    const base = await apiUrl();
    const url = buildEndpointUrl(base, `/api/models/${encodeURIComponent(modelId)}`);
    const response = await fetch(url, {
      method: "DELETE",
      headers: {
        ...SessionService.getSessionHeaders(),
      },
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: "Failed to delete model" }));
      throw new Error(err.error || "Failed to delete model");
    }
  }

  static async createProject(name: string, organism: string): Promise<ProjectItem> {
    const base = await apiUrl();
    const url = buildEndpointUrl(base, "/api/projects");
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...SessionService.getSessionHeaders(),
      },
      body: JSON.stringify({ name, organism }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: "Failed to create project" }));
      throw new Error(err.error || "Failed to create project");
    }
    const data = await response.json();
    return data.project;
  }

  static async listProjects(): Promise<ProjectItem[]> {
    const base = await apiUrl();
    const url = buildEndpointUrl(base, "/api/projects");
    const response = await fetch(url, {
      headers: {
        ...SessionService.getSessionHeaders(),
      },
    });
    if (!response.ok) {
      throw new Error("Failed to fetch projects");
    }
    const data = await response.json();
    return data.projects || [];
  }

  static async submitTrain(projectId: string, dataset: any, config: any): Promise<{ job_id: string }> {
    const base = await apiUrl();
    const url = buildEndpointUrl(base, "/api/train");
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...SessionService.getSessionHeaders(),
      },
      body: JSON.stringify({ project_id: projectId, dataset, config }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: "Failed to submit training" }));
      throw new Error(err.error || "Failed to submit training");
    }
    return response.json();
  }

  static async getTrainJobStatus(jobId: string): Promise<TrainJobStatusResult> {
    const base = await apiUrl();
    const url = buildEndpointUrl(base, `/api/train/${encodeURIComponent(jobId)}`);
    const response = await fetch(url, {
      headers: {
        ...SessionService.getSessionHeaders(),
      },
    });
    if (!response.ok) {
      throw new Error("Failed to fetch train job status");
    }
    return response.json();
  }

  static async deriveBoxes(tpsContent: string, padding: number): Promise<DeriveBoxesResult> {
    const base = await apiUrl();
    const url = buildEndpointUrl(base, "/api/dataset/derive-boxes");
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...SessionService.getSessionHeaders(),
      },
      body: JSON.stringify({ tps_content: tpsContent, padding }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: "Failed to derive boxes" }));
      throw new Error(err.error || "Failed to derive boxes");
    }
    return response.json();
  }

  static async cancelTrainJob(jobId: string): Promise<{ success: boolean; message: string }> {
    const base = await apiUrl();
    const url = buildEndpointUrl(base, `/api/train/${jobId}/cancel`);
    const response = await fetch(url, {
      method: "POST",
      headers: {
        ...SessionService.getSessionHeaders(),
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: "Failed to cancel training job" }));
      throw new Error(err.error || "Failed to cancel training job");
    }
    return response.json();
  }
}

export interface ModelVersionItem {
  id: string;
  project_id: string;
  name: string;
  created_at: string;
  manifest: {
    schema_version: number;
    id: string;
    name: string;
    description: string;
    detector: { artifact: string; geometry: string; confidence: number; iou: number };
    classes: Array<{ id: number; name: string; landmark_schema?: string; predictor?: string; crop_padding?: number }>;
    landmark_schemas: Record<string, { points: string[] }>;
    evaluation?: Record<string, any>;
  };
}

export interface ProjectItem {
  id: string;
  name: string;
  organism: string;
  created_at: string;
}

export interface TrainJobStatusResult {
  success: boolean;
  job_id: string;
  status: string;
  stage: string;
  progress: number;
  metrics: Record<string, any>;
}

export interface DeriveBoxesResult {
  success: boolean;
  padding: number;
  images: Array<{
    image_id: string;
    file_path: string;
    width: number;
    height: number;
    objects: Array<{
      object_id: string;
      class_name: string;
      obb: number[];
      landmarks: Array<{ name: string; x: number; y: number }>;
      specimen_id?: string;
    }>;
  }>;
  total_images: number;
  total_objects: number;
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
  test_accuracy?: number | null;
}
