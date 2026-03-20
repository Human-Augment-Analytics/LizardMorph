interface ElectronAPI {
  isElectron: boolean;
  getBackendPort: () => Promise<number>;
}

interface Window {
  electronAPI?: ElectronAPI;
}
