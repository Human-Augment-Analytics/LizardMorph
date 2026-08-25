function hasDom(): boolean {
  return typeof window !== "undefined";
}

function isTauriRuntime(): boolean {
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

function fallbackApiUrl(): string {
  if (isTauriRuntime()) {
    return "http://127.0.0.1:3005";
  }
  return import.meta.env.VITE_API_URL || "/api";
}

async function resolveApiUrl(): Promise<string> {
  if (hasDom() && window.electronAPI?.isElectron) {
    try {
      const port = await window.electronAPI.getBackendPort();
      return `http://127.0.0.1:${port}`;
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
