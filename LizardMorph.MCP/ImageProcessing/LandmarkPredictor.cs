using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using MathNet.Numerics.Statistics;

namespace LizardMorph.MCP.ImageProcessing
{
    /// <summary>
    /// Landmark predictor that supports both dlib models and fallback dummy predictions
    /// </summary>
    public class LandmarkPredictor : IDisposable
    {
        private readonly string _modelPath;
        private readonly DlibLandmarkPredictor? _dlibPredictor;
        private readonly bool _useDlibModel;
        private bool _disposed = false;

        public LandmarkPredictor(string modelPath)
        {
            _modelPath = modelPath;

            // Check if we have a dlib .dat model
            if (File.Exists(modelPath) && Path.GetExtension(modelPath).ToLower() == ".dat")
            {
                _dlibPredictor = new DlibLandmarkPredictor(modelPath);
                _useDlibModel = _dlibPredictor.IsModelLoaded;
            }
            else
            {
                _dlibPredictor = null;
                _useDlibModel = false;
            }
        }

        /// <summary>
        /// Predict landmarks using dlib model if available, otherwise fallback to dummy predictions
        /// </summary>
        public ShapeDetectionResult PredictLandmarks(Image<Rgb24> image, Rectangle rect)
        {
            if (_useDlibModel && _dlibPredictor != null)
            {
                // Use the dlib model for real predictions
                return _dlibPredictor.PredictLandmarks(image, rect);
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

            // If we have a dlib model, we could test it to get the actual landmark count
            if (_dlibPredictor != null)
            {
                // For now, return default - in a real implementation, 
                // you could create a small test image and count the returned landmarks
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
            if (_useDlibModel && _dlibPredictor != null)
            {
                return _dlibPredictor.GetModelInfo();
            }
            else
            {
                return $"No dlib model found or failed to load.\n" +
                       $"Model path: {_modelPath}\n" +
                       $"Using dummy landmarks for demonstration.\n" +
                       $"Note: Ensure the .dat file exists and is a valid dlib shape predictor model.";
            }
        }

        /// <summary>
        /// Check if a real model is loaded
        /// </summary>
        public bool IsRealModelLoaded => _useDlibModel;

        public void Dispose()
        {
            if (!_disposed)
            {
                _dlibPredictor?.Dispose();
                _disposed = true;
            }
        }
    }
}
