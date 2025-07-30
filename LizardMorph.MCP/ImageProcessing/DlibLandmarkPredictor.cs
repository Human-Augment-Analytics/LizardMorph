using DlibDotNet;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Runtime.InteropServices;

namespace LizardMorph.MCP.ImageProcessing
{
    /// <summary>
    /// Landmark predictor using DlibDotNet to directly work with .dat files
    /// </summary>
    public class DlibLandmarkPredictor : IDisposable
    {
        private readonly ShapePredictor? _shapePredictor;
        private readonly FrontalFaceDetector? _faceDetector;
        private readonly bool _isModelLoaded;
        private bool _disposed = false;

        public DlibLandmarkPredictor(string modelPath)
        {
            try
            {
                if (File.Exists(modelPath))
                {
                    _shapePredictor = ShapePredictor.Deserialize(modelPath);
                    _faceDetector = Dlib.GetFrontalFaceDetector();
                    _isModelLoaded = true;
                }
                else
                {
                    _isModelLoaded = false;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load dlib model: {ex.Message}");
                _isModelLoaded = false;
            }
        }

        public bool IsModelLoaded => _isModelLoaded;

        /// <summary>
        /// Predict landmarks using the dlib model
        /// </summary>
        public ShapeDetectionResult PredictLandmarks(Image<Rgb24> image, Rectangle? boundingBox = null)
        {
            if (!_isModelLoaded || _shapePredictor == null)
            {
                throw new InvalidOperationException("Dlib model is not loaded");
            }

            try
            {
                // Convert ImageSharp image to dlib format
                using var dlibImage = ConvertImageSharpToDlib(image);

                Rectangle detectionRect;
                if (boundingBox.HasValue)
                {
                    // Use provided bounding box
                    detectionRect = boundingBox.Value;
                }
                else
                {
                    // Use the full image as bounding box for lizard images
                    // (since we're not detecting faces but analyzing the entire specimen)
                    detectionRect = new Rectangle(0, 0, image.Width, image.Height);
                }

                // Convert to dlib rectangle
                var dlibRect = new DlibDotNet.Rectangle(
                    detectionRect.Left,
                    detectionRect.Top,
                    detectionRect.Right,
                    detectionRect.Bottom
                );

                // Predict landmarks
                using var shape = _shapePredictor.Detect(dlibImage, dlibRect);

                // Convert dlib landmarks to our format
                var landmarks = new LandmarkPoint[shape.Parts];
                for (uint i = 0; i < shape.Parts; i++)
                {
                    var point = shape.GetPart(i);
                    landmarks[i] = new LandmarkPoint(point.X, point.Y);
                }

                return new ShapeDetectionResult(landmarks);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to predict landmarks: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Detect bounding boxes in the image (useful for automatic region detection)
        /// </summary>
        public Rectangle[] DetectRegions(Image<Rgb24> image)
        {
            if (!_isModelLoaded || _faceDetector == null)
            {
                // Return full image as single region if no detector is available
                return new[] { new Rectangle(0, 0, image.Width, image.Height) };
            }

            try
            {
                using var dlibImage = ConvertImageSharpToDlib(image);
                var detections = _faceDetector.Operator(dlibImage);

                var rectangles = new Rectangle[detections.Length];
                for (int i = 0; i < detections.Length; i++)
                {
                    var detection = detections[i];
                    rectangles[i] = new Rectangle(
                        detection.Left,
                        detection.Top,
                        detection.Right - detection.Left,
                        detection.Bottom - detection.Top
                    );
                }

                // If no detections found, return full image
                if (rectangles.Length == 0)
                {
                    rectangles = new[] { new Rectangle(0, 0, image.Width, image.Height) };
                }

                return rectangles;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Detection failed: {ex.Message}");
                return new[] { new Rectangle(0, 0, image.Width, image.Height) };
            }
        }

        /// <summary>
        /// Convert ImageSharp image to dlib format
        /// </summary>
        private Array2D<RgbPixel> ConvertImageSharpToDlib(Image<Rgb24> image)
        {
            var dlibImage = new Array2D<RgbPixel>(image.Height, image.Width);

            // Extract pixel data from ImageSharp
            var pixelData = new byte[image.Width * image.Height * 3];
            image.CopyPixelDataTo(pixelData);

            // Convert to dlib format
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    int pixelIndex = (y * image.Width + x) * 3;
                    var pixel = new RgbPixel
                    {
                        Red = pixelData[pixelIndex],
                        Green = pixelData[pixelIndex + 1],
                        Blue = pixelData[pixelIndex + 2]
                    };
                    dlibImage[y][x] = pixel;
                }
            }

            return dlibImage;
        }

        /// <summary>
        /// Get information about the loaded model
        /// </summary>
        public string GetModelInfo()
        {
            if (_isModelLoaded)
            {
                return "Dlib shape predictor model loaded successfully\n" +
                       $"Model supports landmark prediction from .dat format";
            }
            else
            {
                return "Dlib model is not loaded. Check if the .dat file exists and is valid.";
            }
        }

        /// <summary>
        /// Process image with automatic region detection and landmark prediction
        /// </summary>
        public ShapeDetectionResult ProcessImage(Image<Rgb24> image)
        {
            // For lizard X-ray images, we typically want to analyze the entire specimen
            // rather than detecting specific regions, so we use the full image bounds
            var fullImageRect = new Rectangle(0, 0, image.Width, image.Height);
            return PredictLandmarks(image, fullImageRect);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _shapePredictor?.Dispose();
                _faceDetector?.Dispose();
                _disposed = true;
            }
        }
    }
}
