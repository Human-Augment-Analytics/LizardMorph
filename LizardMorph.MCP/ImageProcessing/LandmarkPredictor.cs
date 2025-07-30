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
        /// Predict landmarks using dlib model - requires a valid model to be loaded
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
                // Throw an error if no valid model is available
                throw new InvalidOperationException(
                    $"No valid dlib model loaded. Please ensure a valid .dat predictor file exists at: {_modelPath}");
            }
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
                       $"Error: A valid .dat file is required for processing.\n" +
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
