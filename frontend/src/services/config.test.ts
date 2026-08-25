import { afterEach, describe, expect, it, vi } from "vitest";

describe("api config", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it("loads outside a browser runtime and falls back to the relative api path", async () => {
    vi.stubEnv("VITE_API_URL", "");
    vi.resetModules();

    const config = await import("./config");

    expect(config.API_URL).toBe("/api");
    await expect(config.getApiUrl()).resolves.toBe("/api");
  });

  it("honours a configured api url outside a browser runtime", async () => {
    vi.stubEnv("VITE_API_URL", "http://backend.test:3005");
    vi.resetModules();

    const config = await import("./config");

    expect(config.API_URL).toBe("http://backend.test:3005");
    await expect(config.getApiUrl()).resolves.toBe("http://backend.test:3005");
  });
});
