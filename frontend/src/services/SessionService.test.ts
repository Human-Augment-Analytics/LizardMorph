import { describe, expect, it } from "vitest";

describe("session persistence without a browser runtime", () => {
  it("loads and reports no cached session instead of throwing", async () => {
    const { SessionService } = await import("./SessionService");

    expect(SessionService.getStorageType()).toBe("localStorage");
    expect(SessionService.getSessionId()).toBeNull();
    expect(SessionService.hasActiveSession()).toBe(false);
    expect(SessionService.isCachedSessionFresh()).toBe(false);
  });

  it("clears a session without touching absent storage", async () => {
    const { SessionService } = await import("./SessionService");

    expect(() => SessionService.clearSession()).not.toThrow();
    expect(SessionService.getSessionId()).toBeNull();
  });

  it("exposes session headers with no stored id", async () => {
    const { SessionService } = await import("./SessionService");

    expect(SessionService.getSessionHeaders()).not.toHaveProperty("X-Session-ID");
  });
});
