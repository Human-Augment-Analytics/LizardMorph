export function isDesktopRuntime(): boolean {
  const electronAPI = window.electronAPI;
  return Boolean(electronAPI?.isElectron) ||
    Boolean(window.__TAURI__) ||
    Boolean(window.__TAURI_INTERNALS__) ||
    window.location.protocol === "tauri:" ||
    window.location.hostname === "tauri.localhost" ||
    window.location.protocol === "file:";
}

async function resolveApiUrl(): Promise<string> {
  const electronAPI = window.electronAPI;
  const isElectron = Boolean(electronAPI?.isElectron);
  const isTauri = isDesktopRuntime() && !isElectron;

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

export async function fetchWithBackendRetry(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  // The bundled PyInstaller backend can take over a minute to extract on a
  // cold macOS launch. Keep the desktop UI in its loading state long enough
  // for that first boot instead of surfacing a false connection error.
  const attempts = isDesktopRuntime() ? 360 : 1;
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await fetch(input, init);
    } catch (error: unknown) {
      lastError = error;
      if (attempt + 1 < attempts) {
        await new Promise((resolve) => window.setTimeout(resolve, 500));
      }
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Backend is unavailable");
}
