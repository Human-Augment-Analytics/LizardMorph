function hasDom(): boolean {
  return typeof window !== "undefined";
}

export function isDesktopRuntime(): boolean {
  if (!hasDom()) {
    return false;
  }
  const electronAPI = window.electronAPI;
  const runtimeWindow = window as typeof window & { __TAURI__?: unknown; __TAURI_INTERNALS__?: unknown };
  return (
    Boolean(electronAPI?.isElectron) ||
    Boolean(runtimeWindow.__TAURI__) ||
    Boolean(runtimeWindow.__TAURI_INTERNALS__) ||
    window.location.protocol === "tauri:" ||
    window.location.hostname === "tauri.localhost" ||
    window.location.protocol === "file:"
  );
}

export function isTauriRuntime(): boolean {
  if (!hasDom()) {
    return false;
  }
  const runtimeWindow = window as typeof window & { __TAURI__?: unknown };
  return (
    Boolean(runtimeWindow.__TAURI__) ||
    window.location.hostname === "tauri.localhost" ||
    window.location.protocol === "tauri:"
  );
}

export function fallbackApiUrl(): string {
  if (isTauriRuntime()) {
    return "http://127.0.0.1:3005";
  }
  return import.meta.env.VITE_API_URL || "/api";
}

async function resolveApiUrl(): Promise<string> {
  if (hasDom() && window.electronAPI?.isElectron) {
    try {
      const port = await window.electronAPI.getBackendPort();
      if (port) return `http://127.0.0.1:${port}`;
    } catch {
      // fallback
    }
  }
  return fallbackApiUrl();
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
