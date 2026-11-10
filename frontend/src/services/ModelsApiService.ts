import { getApiUrl } from "./config";

export function getModelApiUrl(): string | null {
  const url = (import.meta.env.VITE_MODEL_API_URL ?? "").trim().replace(/\/+$/, "");
  return url || null;
}
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
  listModels: async () => {
    const url = getModelApiUrl();
    if (!url) throw new Error("Research models are not configured (VITE_MODEL_API_URL)");
    return read<{ models: ModelVersionItem[] }>(await fetch(`${url}/models`));
  },
  importBundle: async (name: string, version: string) => {
    const base = (await getApiUrl()).replace(/\/+$/, "");
    const path = base.endsWith("/api") ? "/registry/import" : "/api/registry/import";
    return read<{ success: boolean; model: ImportedModel }>(await fetch(`${base}${path}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, version }),
    }));
  },
};
