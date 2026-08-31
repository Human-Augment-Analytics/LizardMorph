import { afterEach, describe, expect, it, vi } from "vitest";

const TAURI_API_URL = "http://127.0.0.1:3005";

describe("api config", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  it("loads outside a browser runtime and falls back to the relative api path", async () => {
    vi.stubEnv("VITE_API_URL", "");
    vi.resetModules();

    const config = await import("./config");

    await expect(config.getApiUrl()).resolves.toBe("/api");
  });

  it("honours a configured api url outside a browser runtime", async () => {
    vi.stubEnv("VITE_API_URL", "http://backend.test:3005");
    vi.resetModules();

    const config = await import("./config");

    await expect(config.getApiUrl()).resolves.toBe("http://backend.test:3005");
  });

  it("routes the tauri webview at the sidecar port", async () => {
    vi.stubEnv("VITE_API_URL", "http://backend.test:3005");
    vi.stubGlobal("window", {
      location: { protocol: "tauri:", hostname: "localhost" },
    });
    vi.resetModules();

    const config = await import("./config");

    await expect(config.getApiUrl()).resolves.toBe(TAURI_API_URL);
  });

  it("routes the tauri webview at the sidecar port on its http origin", async () => {
    vi.stubEnv("VITE_API_URL", "http://backend.test:3005");
    vi.stubGlobal("window", {
      location: { protocol: "http:", hostname: "tauri.localhost" },
    });
    vi.resetModules();

    const config = await import("./config");

    await expect(config.getApiUrl()).resolves.toBe(TAURI_API_URL);
  });

  it("routes the tauri webview at the sidecar port when only the runtime global is present", async () => {
    vi.stubEnv("VITE_API_URL", "http://backend.test:3005");
    vi.stubGlobal("window", {
      __TAURI__: {},
      location: { protocol: "https:", hostname: "example.test" },
    });
    vi.resetModules();

    const config = await import("./config");

    await expect(config.getApiUrl()).resolves.toBe(TAURI_API_URL);
  });
});
