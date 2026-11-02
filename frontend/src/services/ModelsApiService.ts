import { getApiUrl } from "./config";

const MODEL_API_URL = import.meta.env.VITE_MODEL_API_URL || "/model-api";
export interface ModelVersionItem {
  name: string;
  version: string;
  aliases: string[];
  tags: Record<string, string>;
}
export interface ImportedModel {
  id: string;
  name: string;
  registry: { name: string; version: string; bundle_sha256: string };
}
async function read<T>(response: Response): Promise<T> {
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}
export const ModelsApiService = {
  listModels: async () => read<{ models: ModelVersionItem[] }>(await fetch(`${MODEL_API_URL}/models`)),
  importBundle: async (name: string, version: string) => {
    const base = (await getApiUrl()).replace(/\/+$/, "");
    const path = base.endsWith("/api") ? "/registry/import" : "/api/registry/import";
    return read<{ success: boolean; model: ImportedModel }>(await fetch(`${base}${path}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, version }),
    }));
  },
};
