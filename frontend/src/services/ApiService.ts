import type { ImageSetResponse } from "../models/ImageSetResponse";
import type { AnnotationsData } from "../models/AnnotationsData";
import type { ImageSet } from "../models/ImageSet";

const BASE_URL = "";

export class ApiService {
  static async uploadMultipleImages(files: File[]): Promise<AnnotationsData[]> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append("image", file);
    });
    const res = await fetch(`${BASE_URL}/data`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const errorResult = await res.json();
      throw new Error(errorResult.error ?? "Failed to process images");
    }
    return res.json() as Promise<AnnotationsData[]>;
  }

  static async fetchImageSet(imageFilename: string): Promise<ImageSet> {
    const res = await fetch(
      `${BASE_URL}/image?image_filename=${encodeURIComponent(imageFilename)}`,
      {
        method: "POST",
        headers: {
          "Access-Control-Allow-Origin": "*",
        },
      }
    );
    if (!res.ok) throw new Error("Image set fetch failed");

    const result: ImageSetResponse & { error?: string } = await res.json();
    if (result.error) {
      throw new Error(result.error);
    }

    const fileExtension = imageFilename.split(".").pop()?.toLowerCase() ?? "";
    let mimeType: string;
    if (fileExtension === "png") {
      mimeType = "image/png";
    } else if (fileExtension === "gif") {
      mimeType = "image/gif";
    } else {
      mimeType = "image/jpeg";
    }

    return {
      original: `data:${mimeType};base64,${result.image3}`,
      inverted: `data:${mimeType};base64,${result.image2}`,
      color_contrasted: `data:${mimeType};base64,${result.image1}`,
    };
  }

  static async fetchUploadedFiles(): Promise<string[]> {
    const res = await fetch(`${BASE_URL}/list_uploads`, {
      method: "GET",
    });
    if (!res.ok) throw new Error("Failed to fetch uploaded files");
    return res.json();
  }

  static async processExistingImage(
    filename: string
  ): Promise<AnnotationsData> {
    const res = await fetch(
      `${BASE_URL}/process_existing?filename=${encodeURIComponent(filename)}`,
      {
        method: "POST",
      }
    );
    if (!res.ok) throw new Error("Failed to process existing image");
    return res.json();
  }

  static async saveAnnotations(
    payload: AnnotationsData
  ): Promise<{ success: boolean }> {
    const res = await fetch(`${BASE_URL}/save_annotations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Failed to save annotations");
    return res.json();
  }

  static async downloadAnnotatedImage(imageUrl: string): Promise<Blob> {
    const res = await fetch(imageUrl);
    if (!res.ok) throw new Error("Failed to fetch annotated image");
    return res.blob();
  }

  static async exportScatterData(payload: {
    coords: { x: number; y: number }[];
    name: string;
  }): Promise<{ image_urls?: string[] }> {
    const res = await fetch(`${BASE_URL}/endpoint`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Failed to export scatter data");
    return res.json();
  }
}
