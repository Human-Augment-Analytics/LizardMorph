const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const { startBackend, stopBackend } = require("./python-backend");

let mainWindow;
let backendProc;
let backendPort;

const isDev = !app.isPackaged;

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadURL(
    "data:text/html,<html><body style='display:flex;justify-content:center;align-items:center;height:100vh;font-family:system-ui;background:%23f5f5f5'><div style='text-align:center'><h1>LizardMorph</h1><p>Starting backend server...</p></div></body></html>"
  );

  try {
    const backend = await startBackend(isDev);
    backendProc = backend.proc;
    backendPort = backend.port;

    backendProc.on("exit", (code) => {
      if (code !== 0 && code !== null && mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.loadURL(
          `data:text/html,<html><body style='padding:40px;font-family:system-ui'>` +
          `<h1>Backend Crashed</h1>` +
          `<p>The backend process exited with code ${code}.</p>` +
          `<p>Please restart LizardMorph.</p>` +
          `</body></html>`
        );
      }
    });

    ipcMain.handle("get-backend-port", () => backendPort);

    if (isDev) {
      mainWindow.loadURL("http://localhost:5173");
    } else {
      const frontendPath = path.join(__dirname, "frontend", "index.html");
      mainWindow.loadFile(frontendPath);
    }
  } catch (err) {
    mainWindow.loadURL(
      `data:text/html,<html><body style='padding:40px;font-family:system-ui'><h1>Startup Error</h1><pre>${err.message}</pre></body></html>`
    );
  }
}

app.whenReady().then(createWindow);

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.on("window-all-closed", () => {
  stopBackend(backendProc);
  app.quit();
});

app.on("before-quit", () => {
  stopBackend(backendProc);
});
