const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const { startBackend, stopBackend } = require("./python-backend");

// File logger — writes to ~/Library/Logs/LizardMorph/main.log
const logDir = path.join(app.getPath("home"), "Library", "Logs", "LizardMorph");
fs.mkdirSync(logDir, { recursive: true });
const logFile = path.join(logDir, "main.log");
const logStream = fs.createWriteStream(logFile, { flags: "a" });

function log(msg) {
  const line = `[${new Date().toISOString()}] ${msg}`;
  console.log(line);
  logStream.write(line + "\n");
}

let mainWindow;
let backendProc;
let backendPort;
let backendLogs = [];

const isDev = !app.isPackaged;
log(`App starting. isDev=${isDev}, isPackaged=${app.isPackaged}`);

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      // Allow fetch() for WASM files when loading from file:// protocol
      allowFileAccessFromFileURLs: !isDev,
      // Disable web security in packaged build so data: URIs work in SVG <image> elements
      // when loaded from file:// protocol (needed for base64 image display)
      webSecurity: isDev,
    },
  });

  mainWindow.loadURL(
    "data:text/html,<html><body style='display:flex;justify-content:center;align-items:center;height:100vh;font-family:system-ui;background:%23f5f5f5'><div style='text-align:center'><h1>LizardMorph</h1><p>Starting backend server...</p></div></body></html>"
  );

  try {
    log(`Starting backend (isDev=${isDev})...`);
    const backend = await startBackend(isDev, log);
    backendProc = backend.proc;
    backendPort = backend.port;
    backendLogs = backend.logs;
    log(`Backend ready on port ${backendPort}`);

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
      log(`Loading frontend from: ${frontendPath}`);
      mainWindow.loadFile(frontendPath).catch((err) => {
        log(`Frontend load error: ${err.message}`);
      });
      // Cmd+Option+I opens devtools
    }

    mainWindow.webContents.on("console-message", (event, level, message) => {
      log(`[renderer] ${message}`);
    });
  } catch (err) {
    log(`Startup error: ${err.message}`);
    log(`Backend logs: ${backendLogs.join(" | ")}`);
    const logs = backendLogs.join("\n").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    mainWindow.loadURL(
      `data:text/html,<html><body style='padding:40px;font-family:system-ui'><h1>Startup Error</h1><pre>${err.message}</pre><h2>Backend Logs</h2><pre style='max-height:400px;overflow:auto;background:%23f0f0f0;padding:10px'>${logs || "No output captured"}</pre></body></html>`
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
