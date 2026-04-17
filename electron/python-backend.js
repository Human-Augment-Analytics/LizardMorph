const { spawn } = require("child_process");
const path = require("path");
const net = require("net");
const os = require("os");

function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(0, "127.0.0.1", () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
    server.on("error", reject);
  });
}

function waitForServer(port, timeoutMs = 30000, onRetry = null) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    function tryConnect() {
      const elapsed = Date.now() - start;
      if (elapsed > timeoutMs) {
        return reject(new Error(`Backend did not start within ${timeoutMs / 1000}s`));
      }
      if (onRetry) onRetry(elapsed);
      const req = require("http").get(`http://127.0.0.1:${port}/health`, (res) => {
        if (res.statusCode === 200) {
          resolve();
        } else {
          setTimeout(tryConnect, 500);
        }
      });
      req.on("error", () => setTimeout(tryConnect, 500));
    }
    tryConnect();
  });
}

async function startBackend(isDev, log = console.log) {
  const port = await findFreePort();

  let proc;
  const isWin = process.platform === "win32";

  if (isDev) {
    const backendDir = path.join(__dirname, "..", "backend");
    // Use the Python from PYTHON_PATH env var, or try to find conda env's Python
    const defaultCondaEnv = isWin
      ? path.join(os.homedir(), "miniconda3", "envs", "lizard", "python.exe")
      : path.join(os.homedir(), "miniconda3", "envs", "lizard", "bin", "python");
    const pythonPath = process.env.PYTHON_PATH || defaultCondaEnv;

    proc = spawn(pythonPath, ["app.py"], {
      cwd: backendDir,
      env: {
        ...process.env,
        API_PORT: String(port),
        PYTHONUNBUFFERED: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
  } else {
    const resourcesPath = process.resourcesPath;
    const exeName = isWin ? "app.exe" : "app";
    const exePath = path.join(resourcesPath, "backend", exeName);
    log(`[backend] resourcesPath: ${resourcesPath}`);
    log(`[backend] exePath: ${exePath}`);
    log(`[backend] exists: ${require("fs").existsSync(exePath)}`);
    // Log the actual resolved path (check for app translocation)
    const fs = require("fs");
    const resolvedExe = fs.realpathSync(exePath);
    log(`[backend] resolvedExe: ${resolvedExe}`);

    proc = spawn(resolvedExe, [], {
      cwd: path.dirname(resolvedExe),
      env: {
        ...process.env,
        API_PORT: String(port),
        PYTHONUNBUFFERED: "1",
        HOME: os.homedir(),
        TMPDIR: os.tmpdir(),
      },
      stdio: ["pipe", "pipe", "pipe"],
    });

    log(`[backend] spawned pid=${proc.pid}`);
  }

  const logs = [];

  proc.on("error", (err) => {
    console.error(`[backend] spawn error: ${err.message}`);
    logs.push(`Spawn error: ${err.message}`);
  });

  proc.stdout.on("data", (data) => {
    const line = data.toString().trim();
    console.log(`[backend] ${line}`);
    logs.push(line);
  });

  proc.stderr.on("data", (data) => {
    const line = data.toString().trim();
    console.error(`[backend] ${line}`);
    logs.push(line);
  });

  proc.on("exit", (code) => {
    console.log(`[backend] exited with code ${code}`);
    logs.push(`Process exited with code ${code}`);
  });

  await waitForServer(port);
  console.log(`[backend] ready on port ${port}`);

  return { proc, port, logs };
}

function stopBackend(proc) {
  if (proc && !proc.killed) {
    proc.kill("SIGTERM");
    setTimeout(() => {
      if (!proc.killed) proc.kill("SIGKILL");
    }, 5000);
  }
}

module.exports = { startBackend, stopBackend };
