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

describe("ExportService archive layout", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    vi.resetModules();
  });

  it("keeps specimens sharing a dotted prefix in separate archive entries", async () => {
    const { ApiService } = await import("./ApiService");
    const { ExportService } = await import("./ExportService");
    const JSZip = (await import("jszip")).default;

    vi.spyOn(ApiService, "exportScatterData").mockImplementation(async (payload) => ({
      image_urls: [`/images/01234567/annotated_${payload.name.replace(/\.[^.]+$/, "")}.png`],
    }));
    vi.spyOn(ApiService, "downloadAnnotatedImage").mockImplementation(
      async (imageUrl: string) =>
        new TextEncoder().encode(imageUrl) as unknown as Blob,
    );
    const downloadFile = vi.spyOn(ExportService, "downloadFile").mockResolvedValue();

    await ExportService.exportAllData(
      [
        { name: "LIZ.001.jpg", coords: [{ id: 0, x: 10, y: 12 }], imageSets: { original: "a" } },
        { name: "LIZ.002.jpg", coords: [{ id: 0, x: 30, y: 24 }], imageSets: { original: "b" } },
      ],
      0,
      [{ id: 0, x: 10, y: 12 }],
      [{ id: 0, x: 10, y: 12 }],
      async () => new TextEncoder().encode("overlay") as unknown as Blob,
      [],
      SCALE_SETTINGS,
    );

    const archived = await downloadFile.mock.calls[0][0].arrayBuffer();
    const archive = await JSZip.loadAsync(archived);

    expect(Object.keys(archive.files).sort()).toEqual([
      "LIZ.001.tps",
      "LIZ.002.tps",
      "annotated_LIZ.001.png",
      "annotated_LIZ.002.png",
      "measurements.csv",
    ]);
    await expect(archive.file("LIZ.001.tps")!.async("string")).resolves.toContain(
      "IMAGE=LIZ.001",
    );
    await expect(archive.file("LIZ.002.tps")!.async("string")).resolves.toContain(
      "IMAGE=LIZ.002",
    );
    await expect(
      archive.file("annotated_LIZ.001.png")!.async("string"),
    ).resolves.toBe("/api/images/01234567/annotated_LIZ.001.png");
    await expect(
      archive.file("annotated_LIZ.002.png")!.async("string"),
    ).resolves.toBe("/api/images/01234567/annotated_LIZ.002.png");
  });
});

describe("ExportService measurement CSV", () => {
  const coords = [
    { id: 0, x: 0, y: 0 },
    { id: 1, x: 3, y: 4 },
  ];
  const scale: ScaleSettings = {
    pointAId: 0,
    pointBId: 1,
    value: 10,
    units: "mm",
  };

  it.each([
    ["specimen", "specimen"],
    ["specimen, left", '"specimen, left"'],
    ['specimen "left"', '"specimen ""left"""'],
    ["specimen\nleft", '"specimen\nleft"'],
    ["specimen\rleft", '"specimen\rleft"'],
  ])("preserves text fields containing %j", async (text, encoded) => {
    const { ExportService } = await import("./ExportService");

    const csv = ExportService.createCsvContent(
      [{ id: "m1", label: text, pointAId: 0, pointBId: 1, calculatedDistance: null }],
      { ...scale, units: text },
      [{ name: text, coords }],
    );

    expect(csv).toBe(
      `Measurement,Image,Point A,Point B,Distance,Units\n${encoded},${encoded},1,2,10.000,${encoded}\n`,
    );
  });

  it("exports coincident measured landmarks as zero with a valid scale", async () => {
    const { ExportService } = await import("./ExportService");

    const csv = ExportService.createCsvContent([], scale, [{
      name: "a.png",
      coords: [...coords, { id: 2, x: 0, y: 0 }],
    }]);

    expect(csv).toBe(
      "Measurement,Image,Point A,Point B,Distance,Units\n" +
      "Landmark 1-2,a.png,1,2,10.000,mm\n" +
      "Landmark 1-3,a.png,1,3,0.000,mm\n" +
      "Landmark 2-3,a.png,2,3,10.000,mm\n",
    );
  });

  it("retains N/A when no scale is configured", async () => {
    const { ExportService } = await import("./ExportService");

    expect(ExportService.createCsvContent([], SCALE_SETTINGS, [{ name: "a.png", coords }]))
      .toBe(
        "Measurement,Image,Point A,Point B,Distance,Units\n" +
        "Landmark 1-2,a.png,1,2,N/A,pixels\n",
      );
  });

  it("retains N/A when a coordinate produces an undefined numerical distance", async () => {
    const { ExportService } = await import("./ExportService");

    expect(ExportService.createCsvContent([], scale, [{
      name: "a.png",
      coords: [coords[0], { ...coords[1], x: NaN }],
    }])).toBe(
      "Measurement,Image,Point A,Point B,Distance,Units\n" +
      "Landmark 1-2,a.png,1,2,N/A,mm\n",
    );
  });
});
