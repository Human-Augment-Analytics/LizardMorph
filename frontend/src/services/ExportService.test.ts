import { afterEach, describe, expect, it, vi } from "vitest";

import type { ScaleSettings } from "../models/ScaleSettings";

const SCALE_SETTINGS: ScaleSettings = {
  pointAId: null,
  pointBId: null,
  value: null,
  units: "pixels",
};

async function exportOneImage() {
  const { ApiService } = await import("./ApiService");
  const { ExportService } = await import("./ExportService");

  vi.spyOn(ApiService, "exportScatterData").mockResolvedValue({
    image_urls: ["/images/01234567/output_1.png"],
  });
  const downloadAnnotatedImage = vi
    .spyOn(ApiService, "downloadAnnotatedImage")
    .mockResolvedValue(new Uint8Array([1, 2, 3]) as unknown as Blob);
  vi.spyOn(ExportService, "downloadFile").mockResolvedValue();

  await ExportService.exportAllData(
    [{ name: "a.png", coords: [{ id: 0, x: 1, y: 2 }], imageSets: { original: "a" } }],
    0,
    [{ id: 0, x: 1, y: 2 }],
    [{ id: 0, x: 1, y: 2 }],
    async () => new Uint8Array([4, 5, 6]) as unknown as Blob,
    [],
    SCALE_SETTINGS,
  );

  return downloadAnnotatedImage;
}

describe("ExportService annotated image download", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    vi.resetModules();
  });

  it("fetches the annotated image from the Electron backend port", async () => {
    vi.stubGlobal("window", {
      location: { protocol: "http:", hostname: "localhost" },
      electronAPI: { isElectron: true, getBackendPort: async () => 7777 },
    });
    vi.resetModules();

    const downloadAnnotatedImage = await exportOneImage();

    expect(downloadAnnotatedImage).toHaveBeenCalledWith(
      "http://127.0.0.1:7777/images/01234567/output_1.png",
    );
  });

  it("fetches the annotated image from the Tauri backend base", async () => {
    vi.stubGlobal("window", {
      location: { protocol: "tauri:", hostname: "localhost" },
    });
    vi.resetModules();

    const downloadAnnotatedImage = await exportOneImage();

    expect(downloadAnnotatedImage).toHaveBeenCalledWith(
      "http://127.0.0.1:3005/images/01234567/output_1.png",
    );
  });
});
