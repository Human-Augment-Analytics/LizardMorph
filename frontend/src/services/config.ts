async function resolveApiUrl(): Promise<string> {
  if (window.electronAPI?.isElectron) {
    try {
      const port = await window.electronAPI.getBackendPort();
      return `http://127.0.0.1:${port}`;
    } catch {
      // fallback
    }
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

// Synchronous export for non-Electron web mode (backwards compat)
export const API_URL = import.meta.env.VITE_API_URL || "/api";
