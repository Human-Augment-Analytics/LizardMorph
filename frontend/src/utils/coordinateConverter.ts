
// Coordinate conversion utility for ONNX to YOLO preprocessing
export function convertOnnxToYoloCoords(
    onnxBbox: number[],
    originalWidth: number,
    originalHeight: number,
    onnxInputSize: number = 640,
    yoloTargetSize: number = 1024
): number[] {
    const [x1, y1, x2, y2] = onnxBbox;
    
    // Step 1: Convert from ONNX letterbox to original image coordinates
    const onnxScale = Math.min(onnxInputSize / originalWidth, onnxInputSize / originalHeight);
    const onnxScaledWidth = originalWidth * onnxScale;
    const onnxScaledHeight = originalHeight * onnxScale;
    const onnxPadX = (onnxInputSize - onnxScaledWidth) / 2;
    const onnxPadY = (onnxInputSize - onnxScaledHeight) / 2;
    
    const origX1 = (x1 - onnxPadX) / onnxScale;
    const origY1 = (y1 - onnxPadY) / onnxScale;
    const origX2 = (x2 - onnxPadX) / onnxScale;
    const origY2 = (y2 - onnxPadY) / onnxScale;
    
    // Step 2: Convert from original coordinates to YOLO direct resize coordinates
    const yoloScale = Math.min(yoloTargetSize / originalWidth, yoloTargetSize / originalHeight);
    
    const yoloX1 = origX1 * yoloScale;
    const yoloY1 = origY1 * yoloScale;
    const yoloX2 = origX2 * yoloScale;
    const yoloY2 = origY2 * yoloScale;
    
    return [yoloX1, yoloY1, yoloX2, yoloY2];
}
