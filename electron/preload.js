const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  isElectron: true,
  getBackendPort: () => ipcRenderer.invoke("get-backend-port"),
});
