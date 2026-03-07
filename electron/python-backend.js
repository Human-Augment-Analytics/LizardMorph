const { spawn } = require("child_process");
const path = require("path");
const net = require("net");

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

function waitForServer(port, timeoutMs = 60000, onRetry = null) {
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

async function startBackend(isDev) {
  const port = await findFreePort();

  let proc;
  if (isDev) {
    const backendDir = path.join(__dirname, "..", "backend");
    proc = spawn("python", ["app.py"], {
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
    const exePath = path.join(resourcesPath, "backend", "app");
    proc = spawn(exePath, [], {
      env: {
        ...process.env,
        API_PORT: String(port),
        PYTHONUNBUFFERED: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
  }

  proc.stdout.on("data", (data) => {
    console.log(`[backend] ${data.toString().trim()}`);
  });

  proc.stderr.on("data", (data) => {
    console.error(`[backend] ${data.toString().trim()}`);
  });

  proc.on("exit", (code) => {
    console.log(`[backend] exited with code ${code}`);
  });

  await waitForServer(port);
  console.log(`[backend] ready on port ${port}`);

  return { proc, port };
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
