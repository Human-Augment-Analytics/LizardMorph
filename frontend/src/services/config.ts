async function resolveApiUrl(): Promise<string> {
  const electronAPI = window.electronAPI;
  const isElectron = Boolean(electronAPI?.isElectron);
  const isTauri =
    Boolean((window as any).__TAURI__) ||
    Boolean((window as any).__TAURI_INTERNALS__) ||
    window.location.protocol === "tauri:" ||
    window.location.hostname === "tauri.localhost" ||
    window.location.protocol === "file:";

  if (isElectron && electronAPI) {
    try {
      const port = await electronAPI.getBackendPort();
      return `http://127.0.0.1:${port}`;
    } catch {
      // fallback
    }
  }

  if (isTauri) {
    const defaultPort = import.meta.env.VITE_API_PORT || "3005";
    return `http://127.0.0.1:${defaultPort}`;
  }

  return import.meta.env.VITE_API_URL || "/api";
}

let _apiUrlPromise: Promise<string> | null = null;

export function getApiUrl(): Promise<string> {
  if (!_apiUrlPromise) {
    _apiUrlPromise = resolveApiUrl();
  }
  return _apiUrlPromise;
}

export const API_URL = import.meta.env.VITE_API_URL || "/api";
