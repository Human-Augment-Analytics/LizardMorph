import { afterEach, expect, it, vi } from "vitest";
import { ModelsApiService } from "./ModelsApiService";
import * as config from "./config";

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

it.each(["/api", "http://127.0.0.1:3005"])("imports the explicit named version through %s", async base => {
  vi.spyOn(config, "getApiUrl").mockResolvedValue(base);
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ model: { id: "pinned" } }) });
  vi.stubGlobal("fetch", fetch);
  await ModelsApiService.importBundle("toepad-vit", "7");
  expect(fetch.mock.calls[0][0]).toBe(base === "/api" ? "/api/registry/import" : `${base}/api/registry/import`);
  expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ name: "toepad-vit", version: "7" });
});

it("surfaces an invalid bundle response without reporting success", async () => {
  vi.spyOn(config, "getApiUrl").mockResolvedValue("/api");
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 400, text: async () => "checksum mismatch" }));
  await expect(ModelsApiService.importBundle("toepad-vit", "7")).rejects.toThrow("checksum mismatch");
});
