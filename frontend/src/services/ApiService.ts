import type { ImageSetResponse } from "../models/ImageSetResponse";
import type { Point } from "../models/Point";
import type { UploadResponse } from "../models/UploadResponse";

const BASE_URL = '/api';

export class ApiService {
  static async uploadImage(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('image', file);
    const res = await fetch(`${BASE_URL}/data`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Upload failed');
    return res.json() as Promise<UploadResponse>;
  }

  static async fetchImageSet(imageFilename: string): Promise<ImageSetResponse> {
    const res = await fetch(`${BASE_URL}/image?image_filename=${encodeURIComponent(imageFilename)}`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Image set fetch failed');
    return res.json() as Promise<ImageSetResponse>;
  }

  static async postScatterData(payload: { name: string; coords: Point[] }): Promise<{ success: boolean }> {
    const res = await fetch(`${BASE_URL}/endpoint`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('Post failed');
    return res.json();
  }

  static async fetchImageBlob(imageFilename: string): Promise<string> {
    const res = await fetch(`${BASE_URL}/image?image_filename=${encodeURIComponent(imageFilename)}`, {
      method: 'POST',
    });
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  }
}
