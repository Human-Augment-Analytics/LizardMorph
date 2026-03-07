const { app, BrowserWindow } = require("electron");
const path = require("path");

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // For now, load a placeholder. Will point to frontend build later.
  mainWindow.loadURL("data:text/html,<h1>LizardMorph Desktop</h1><p>Electron shell works.</p>");
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  app.quit();
});
