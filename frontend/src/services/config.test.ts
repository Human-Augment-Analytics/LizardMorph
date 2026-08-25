import { describe, expect, it } from "vitest";

describe("api config", () => {
  it("loads outside a browser runtime and falls back to the relative api path", async () => {
    const config = await import("./config");

    expect(config.API_URL).toBe("/api");
    await expect(config.getApiUrl()).resolves.toBe("/api");
  });
});
