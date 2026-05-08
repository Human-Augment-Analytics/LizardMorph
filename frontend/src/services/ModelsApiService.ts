const MODEL_API_URL = import.meta.env.VITE_MODEL_API_URL || "http://localhost:8000";
const APP_API_URL = import.meta.env.VITE_API_URL || "http://localhost:3005";

export interface HealthResponse {
  status: string;
  mlflow_reachable: boolean;
}

export interface ExperimentItem {
  experiment_id: string;
  name: string;
  lifecycle_stage: string;
}

export interface RunItem {
  run_id: string;
  run_name: string | null;
  status: string;
  metrics: Record<string, number>;
  params: Record<string, string>;
  tags: Record<string, string>;
}

export interface ModelVersionItem {
  name: string;
  version: string;
  stage: string;
  aliases: string[];
  run_id: string | null;
  source: string | null;
  status: string | null;
  tags: Record<string, string>;
}

export interface PromoteModelRequest {
  alias?: string;
  set_stage?: boolean;
  stage?: "Production" | "Staging" | "Archived" | "None";
  archive_existing_versions?: boolean;
}

export interface PromoteModelResponse {
  name: string;
  version: string;
  alias: string;
  stage: string;
  aliases: string[];
  run_id: string | null;
}

export interface LatestModelResponse {
  version: string;
  trained: string;
  author: string;
  architecture: string;
  task: string;
  config: string;
  dataset: { nc: number; names: string[]; train_images: number; val_images: number };
  training: { epochs: number; batch: number; imgsz: number; patience: number };
  metrics: {
    best_epoch: number;
    epochs_completed: number;
    mAP50: number;
    "mAP50-95": number;
    precision: number;
    recall: number;
  };
  assets: { fp16: string; fp32: string; pt: string };
}

export interface SyncBackendResponse {
  status: string;
  loaded_model: string;
  version: string;
}

async function get<T>(baseUrl: string, path: string): Promise<T> {
  const resp = await fetch(`${baseUrl}${path}`);
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`${resp.status} ${resp.statusText}: ${body}`);
  }
  return resp.json();
}

async function post<T>(baseUrl: string, path: string, body: unknown): Promise<T> {
  const resp = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
  }
  return resp.json();
}

export const ModelsApiService = {
  getHealth: () => get<HealthResponse>(MODEL_API_URL, "/health"),
  listExperiments: () =>
    get<{ experiments: ExperimentItem[] }>(MODEL_API_URL, "/experiments"),
  listRuns: (experimentId: string) =>
    get<{ experiment_id: string; runs: RunItem[] }>(
      MODEL_API_URL,
      `/experiments/${experimentId}/runs`
    ),
  listModels: () =>
    get<{ models: ModelVersionItem[] }>(MODEL_API_URL, "/models"),
  promoteModel: (
    name: string,
    version: string,
    request: PromoteModelRequest = {}
  ) =>
    post<PromoteModelResponse>(
      MODEL_API_URL,
      `/models/${encodeURIComponent(name)}/versions/${encodeURIComponent(version)}/promote`,
      request
    ),
  getLatestModel: () => get<LatestModelResponse>(MODEL_API_URL, "/model/latest"),
  getModelByTag: (tag: string) =>
    get<LatestModelResponse>(MODEL_API_URL, `/model/by-tag/${tag}`),
  syncBackend: () =>
    post<SyncBackendResponse>(APP_API_URL, "/api/sync-model", {}),
};
