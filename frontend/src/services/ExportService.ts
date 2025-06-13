import JSZip from "jszip";
import type { Point } from "../models/Point";
import { ApiService } from "./ApiService";

export class ExportService {
  static createTpsContent(coords: Point[], imageName: string): string {
    let tpsContent = `LM=${coords.length}\n`;
    coords.forEach(point => {
      tpsContent += `${point.x} ${point.y}\n`;
    });
    tpsContent += `IMAGE=${imageName.split('.')[0]}`;
    return tpsContent;
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

    // Trigger the download
    document.body.appendChild(link);
    link.click();

    // Clean up
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
    createImageWithPointsBlob: (index: number, name: string) => Promise<Blob>
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
          // Create TPS file content
          const tpsContent = this.createTpsContent(originalCoords, payload.name);

          // Send data to backend to get annotated image
          const result = await ApiService.exportScatterData(payload);
          console.log(`Processed data for ${payload.name}:`, result);

          let imageBlob: Blob | undefined;

          // Try to get the annotated image if it exists
          if (result.image_urls && result.image_urls.length > 0) {
            const imageUrl = result.image_urls[0].startsWith('http')
              ? result.image_urls[0]
              : `/api${result.image_urls[0].startsWith('/') ? '' : '/'}${result.image_urls[0]}`;

            try {
              imageBlob = await ApiService.downloadAnnotatedImage(imageUrl);
            } catch (imageError) {
              console.warn(`Failed to fetch annotated image for ${payload.name}:`, imageError);
            }
          } else {
            console.warn(`Backend processing failed for ${payload.name}, but continuing with TPS file`);
          }

          // If no annotated image from backend, create one from current visualization
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
          // Still return TPS content even if other processing fails
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

      // Process successful results
      const successfulResults = results
        .filter((result): result is PromiseFulfilledResult<{ name: string; tpsContent: string; imageBlob?: Blob }> =>
          result.status === 'fulfilled')
        .map(result => result.value);

      if (successfulResults.length === 0) {
        throw new Error('No files were processed successfully');
      }

      // Add files to zip
      successfulResults.forEach(result => {
        const baseName = result.name.split('.')[0];
        
        // Add TPS file
        zip.file(`${baseName}.tps`, result.tpsContent);

        // Add annotated image if available
        if (result.imageBlob) {
          const imageExt = result.name.split('.').pop()?.toLowerCase() || 'png';
          zip.file(`annotated_${baseName}.${imageExt}`, result.imageBlob);
        }
      });

      // Generate and download zip file
      const zipBlob = await zip.generateAsync({ type: 'blob' });
      await this.downloadFile(zipBlob, `lizard_analysis_${new Date().toISOString().split('T')[0]}.zip`);

      // Return success/failure information
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