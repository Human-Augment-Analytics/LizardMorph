using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Numerics;
using MathNet.Numerics.Statistics;

namespace LizardMorph.MCP.ImageProcessing
{
    /// <summary>
    /// Landmark predictor that supports both ONNX models and fallback dummy predictions
    /// </summary>
    public class LandmarkPredictor : IDisposable
    {
        private readonly string _modelPath;
        private readonly OnnxLandmarkPredictor? _onnxPredictor;
        private readonly bool _useOnnxModel;
        private bool _disposed = false;

        public LandmarkPredictor(string modelPath)
        {
            _modelPath = modelPath;
            
            // Check if we have an ONNX model
            var onnxModelPath = Path.ChangeExtension(modelPath, ".onnx");
            if (File.Exists(onnxModelPath))
            {
                _onnxPredictor = new OnnxLandmarkPredictor(onnxModelPath);
                _useOnnxModel = _onnxPredictor.IsModelLoaded;
            }
            else
            {
                _onnxPredictor = null;
                _useOnnxModel = false;
            }
        }

        /// <summary>
        /// Predict landmarks using ONNX model if available, otherwise fallback to dummy predictions
        /// </summary>
        public ShapeDetectionResult PredictLandmarks(Image<Rgb24> image, Rectangle rect)
        {
            if (_useOnnxModel && _onnxPredictor != null)
            {
                // Use the ONNX model for real predictions
                return _onnxPredictor.PredictLandmarks(image, rect);
            }
            else
            {
                // Fallback to dummy landmarks for demonstration
                return GenerateDummyLandmarks(rect);
            }
        }

        /// <summary>
        /// Generate dummy landmarks as fallback when no model is available
        /// </summary>
        private ShapeDetectionResult GenerateDummyLandmarks(Rectangle rect)
        {
            // Try to determine number of landmarks from existing data files if available
            int numLandmarks = DetermineLandmarkCount();
            
            var points = new LandmarkPoint[numLandmarks];
            var random = new Random(42); // Fixed seed for consistent results
            
            // Generate landmarks distributed across the rectangle
            // This creates a more realistic distribution than pure random
            for (int i = 0; i < numLandmarks; i++)
            {
                // Create some structure in the landmark placement
                var normalizedIndex = (float)i / numLandmarks;
                
                // Create a rough anatomical distribution for lizard landmarks
                var x = rect.Left + (0.1f + 0.8f * normalizedIndex + random.NextSingle() * 0.1f) * rect.Width;
                var y = rect.Top + (0.2f + 0.6f * (float)Math.Sin(normalizedIndex * Math.PI) + random.NextSingle() * 0.2f) * rect.Height;
                
                points[i] = new LandmarkPoint(x, y);
            }
            
            return new ShapeDetectionResult(points);
        }

        /// <summary>
        /// Try to determine the expected number of landmarks from the model or existing data
        /// </summary>
        private int DetermineLandmarkCount()
        {
            // Default number of landmarks for lizard anatomy
            const int defaultLandmarks = 42; // Typical for lizard morphological studies
            
            // If we have an ONNX model, we could inspect its output shape
            if (_onnxPredictor != null)
            {
                // For now, return default - in a real implementation, 
                // you could inspect the ONNX model metadata
                return defaultLandmarks;
            }
            
            // Try to infer from existing XML files in the same directory
            var modelDir = Path.GetDirectoryName(_modelPath);
            if (!string.IsNullOrEmpty(modelDir))
            {
                var xmlFiles = Directory.GetFiles(modelDir, "*.xml");
                if (xmlFiles.Length > 0)
                {
                    try
                    {
                        // Parse a sample XML file to get landmark count
                        var sampleXml = System.Xml.Linq.XDocument.Load(xmlFiles[0]);
                        var parts = sampleXml.Descendants("part");
                        if (parts.Any())
                        {
                            return parts.Count();
                        }
                    }
                    catch
                    {
                        // Ignore parsing errors, use default
                    }
                }
            }
            
            return defaultLandmarks;
        }

        /// <summary>
        /// Generate multi-scale landmark predictions and compute median
        /// </summary>
        public LandmarkPoint[] PredictMultiScale(Image<Rgb24> image, float[] scales)
        {
            var allLandmarks = new List<LandmarkPoint[]>();
            
            foreach (var scale in scales)
            {
                using var scaledImage = ImageProcessor.ResizeImage(image, scale);
                var rect = new Rectangle(1, 1, 
                    (int)(image.Width * scale) - 1, 
                    (int)(image.Height * scale) - 1);
                
                var result = PredictLandmarks(scaledImage, rect);
                
                // Scale landmarks back to original image size
                var scaledLandmarks = result.Points.Select(p => new LandmarkPoint(p.X / scale, p.Y / scale)).ToArray();
                allLandmarks.Add(scaledLandmarks);
            }
            
            // Compute median coordinates across scales
            if (allLandmarks.Count == 0) return Array.Empty<LandmarkPoint>();
            
            var numPoints = allLandmarks[0].Length;
            var medianLandmarks = new LandmarkPoint[numPoints];
            
            for (int i = 0; i < numPoints; i++)
            {
                var xValues = allLandmarks.Select(landmarks => (double)landmarks[i].X).ToArray();
                var yValues = allLandmarks.Select(landmarks => (double)landmarks[i].Y).ToArray();
                
                var medianX = xValues.Median();
                var medianY = yValues.Median();
                
                medianLandmarks[i] = new LandmarkPoint((float)medianX, (float)medianY);
            }
            
            return medianLandmarks;
        }

        /// <summary>
        /// Get information about the loaded model
        /// </summary>
        public string GetModelInfo()
        {
            if (_useOnnxModel && _onnxPredictor != null)
            {
                return _onnxPredictor.GetModelInfo();
            }
            else
            {
                var onnxPath = Path.ChangeExtension(_modelPath, ".onnx");
                return $"No ONNX model found. Looking for: {onnxPath}\n" +
                       $"Original model path: {_modelPath}\n" +
                       $"Using dummy landmarks for demonstration.";
            }
        }

        /// <summary>
        /// Check if a real model is loaded
        /// </summary>
        public bool IsRealModelLoaded => _useOnnxModel;

        public void Dispose()
        {
            if (!_disposed)
            {
                _onnxPredictor?.Dispose();
                _disposed = true;
            }
        }
    }
}
