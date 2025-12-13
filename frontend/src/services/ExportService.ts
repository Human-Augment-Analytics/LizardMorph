import JSZip from "jszip";
import type { Point } from "../models/Point";
import { ApiService } from "./ApiService";
import type { Measurement } from "../models/Measurement";
import type { ScaleSettings } from "../models/ScaleSettings";
import { API_URL } from "./config";

export class ExportService {
  static createTpsContent(coords: Point[], imageName: string): string {
    let tpsContent = `LM=${coords.length}\n`;
    coords.forEach(point => {
      tpsContent += `${point.x} ${point.y}\n`;
    });
    tpsContent += `IMAGE=${imageName.split('.')[0]}`;
    return tpsContent;
  }

  static createCsvContent(
    measurements: Measurement[],
    scaleSettings: ScaleSettings,
    images: Array<{ name: string; coords: Point[] }>
  ): string {
    let csvContent = "Measurement,Image,Point A,Point B,Distance,Units\n";

    images.forEach((image) => {
      // Generate all pairwise combinations of landmarks
      const coords = image.coords;
      for (let i = 0; i < coords.length; i++) {
        for (let j = i + 1; j < coords.length; j++) {
          const pointA = coords[i];
          const pointB = coords[j];
          
          if (pointA && pointB) {
            const distance = this.calculateDistance(
              pointA,
              pointB,
              scaleSettings,
              coords
            );
            // Check if there's a custom label for this measurement
            const customMeasurement = measurements.find(
              (m) => m.pointAId === pointA.id && m.pointBId === pointB.id
            );
            const label = customMeasurement?.label || `Landmark ${pointA.id}-${pointB.id}`;
            csvContent += `${label},${image.name},${pointA.id},${pointB.id},${distance ? distance.toFixed(3) : "N/A"},${scaleSettings.units}\n`;
          }
        }
      }
    });

    return csvContent;
  }

  static calculateDistance(
    pointA: Point,
    pointB: Point,
    scaleSettings: ScaleSettings,
    coords: Point[]
  ): number | null {
    if (scaleSettings.pointAId === null || scaleSettings.pointAId === undefined ||
      scaleSettings.pointBId === null || scaleSettings.pointBId === undefined ||
      scaleSettings.value === null || scaleSettings.value <= 0) {
      return null;
    }

    const pixelDistance = Math.sqrt(
      Math.pow(pointB.x - pointA.x, 2) + Math.pow(pointB.y - pointA.y, 2)
    );

    const scalePointA = coords.find(
      (p) => p.id === scaleSettings.pointAId
    );
    const scalePointB = coords.find(
      (p) => p.id === scaleSettings.pointBId
    );

    if (!scalePointA || !scalePointB) {
      return null;
    }

    const scalePixelDistance = Math.sqrt(
      Math.pow(scalePointB.x - scalePointA.x, 2) +
        Math.pow(scalePointB.y - scalePointA.y, 2)
    );

    if (scalePixelDistance === 0) {
      return null;
    }
    const realDistance = (pixelDistance / scalePixelDistance) * scaleSettings.value;

    return realDistance;
  }
  static async downloadTpsFile(coords: Point[], imageName: string): Promise<void> {
    try {
      const tpsContent = this.createTpsContent(coords, imageName);
      const blob = new Blob([tpsContent], { type: "text/plain" });
      await this.downloadFile(blob, `${imageName.split(".")[0]}.tps`);
      console.log("TPS file downloaded successfully:", imageName);
    } catch (error) {
      console.error("Error downloading TPS file:", error);
      throw error;
    }
  }
  static async downloadFile(blob: Blob, filename: string): Promise<void> {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  static async exportAllData(
    images: Array<{
      name: string;
      coords: Point[];
      originalCoords?: Point[];
      imageSets: { original: string };
    }>,
    currentIndex: number,
    currentScatterData: Point[],
    originalScatterData: Point[],
    createImageWithPointsBlob: (index: number, name: string) => Promise<Blob>,
    measurements: Measurement[],
    scaleSettings: ScaleSettings
  ): Promise<{ totalFiles: number; failedFiles: number; successfulFiles: number }> {
    const downloadPromises: Promise<{ name: string; tpsContent: string; imageBlob?: Blob }>[] = [];
    const zip = new JSZip();
    console.log(currentScatterData);
    for (let i = 0; i < images.length; i++) {
      const originalCoords =
        i === currentIndex
          ? originalScatterData
          : images[i].originalCoords || images[i].coords;

      const payload = {
        coords: originalCoords,
        name: images[i].name
      };

      const processPromise = (async (): Promise<{ name: string; tpsContent: string; imageBlob?: Blob }> => {
        try {
          const tpsContent = this.createTpsContent(originalCoords, payload.name);

          const result = await ApiService.exportScatterData(payload);
          console.log(`Processed data for ${payload.name}:`, result);

          let imageBlob: Blob | undefined;
          if (result.image_urls && result.image_urls.length > 0) {
            const imageUrl = result.image_urls[0].startsWith('http')
              ? result.image_urls[0]
              : `${API_URL}/${result.image_urls[0].startsWith('/') ? '' : '/'}${result.image_urls[0]}`;

            try {
              imageBlob = await ApiService.downloadAnnotatedImage(imageUrl);
            } catch (imageError) {
              console.warn(`Failed to fetch annotated image for ${payload.name}:`, imageError);
            }
          } else {
            console.warn(`Backend processing failed for ${payload.name}, but continuing with TPS file`);
          }
          if (!imageBlob) {
            try {
              imageBlob = await createImageWithPointsBlob(i, payload.name);
            } catch (overlayError) {
              console.warn(`Failed to create overlay image for ${payload.name}:`, overlayError);
            }
          }

          return {
            name: payload.name,
            tpsContent,
            imageBlob
          };
        } catch (error) {
          console.error(`Error processing ${payload.name}:`, error);
          const tpsContent = this.createTpsContent(originalCoords, payload.name);
          return {
            name: payload.name,
            tpsContent
          };
        }
      })();

      downloadPromises.push(processPromise);
    }

    try {
      const results = await Promise.allSettled(downloadPromises);

      const successfulResults = results
        .filter((result): result is PromiseFulfilledResult<{ name: string; tpsContent: string; imageBlob?: Blob }> =>
          result.status === 'fulfilled')
        .map(result => result.value);

      if (successfulResults.length === 0) {
        throw new Error('No files were processed successfully');
      }

      successfulResults.forEach(result => {
        const baseName = result.name.split('.')[0];
        
        zip.file(`${baseName}.tps`, result.tpsContent);
        if (result.imageBlob) {
          const imageExt = result.name.split('.').pop()?.toLowerCase() || 'png';
          zip.file(`annotated_${baseName}.${imageExt}`, result.imageBlob);
        }
      });
      const allOriginalCoords = images.map((image, i) => i === currentIndex ? originalScatterData : image.originalCoords || image.coords);
      const csvContent = this.createCsvContent(measurements, scaleSettings, images.map((img, i) => ({ name: img.name, coords: allOriginalCoords[i] })));
      zip.file("measurements.csv", csvContent);
      const zipBlob = await zip.generateAsync({ type: 'blob' });
      await this.downloadFile(zipBlob, `lizard_analysis_${new Date().toISOString().split('T')[0]}.zip`);
      const failures = results.filter(r => r.status === 'rejected');
      return {
        totalFiles: images.length,
        failedFiles: failures.length,
        successfulFiles: successfulResults.length
      };
    } catch (error) {
      console.error('Export error:', error);
      throw error;
    }
  }
}
 