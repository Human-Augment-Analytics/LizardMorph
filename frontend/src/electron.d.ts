interface ElectronAPI {
  isElectron: boolean;
  getBackendPort: () => Promise<number>;
}

interface Window {
  electronAPI?: ElectronAPI;
  __TAURI__?: unknown;
  __TAURI_INTERNALS__?: unknown;
}
