using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Numerics.Tensors;

namespace LizardMorph.MCP.ImageProcessing
{
    /// <summary>
    /// Input data structure for ONNX model
    /// </summary>
    public class ImageInput
    {
        [ColumnName("input")]
        [VectorType(1, 3, 224, 224)] // Adjust dimensions based on your model
        public float[] ImageData { get; set; } = Array.Empty<float>();
    }

    /// <summary>
    /// Output data structure for ONNX model
    /// </summary>
    public class LandmarkOutput
    {
        [ColumnName("output")]
        [VectorType(136)] // 68 landmarks * 2 coordinates, adjust based on your model
        public float[] Landmarks { get; set; } = Array.Empty<float>();
    }

    /// <summary>
    /// ONNX-based landmark predictor using ML.NET
    /// </summary>
    public class OnnxLandmarkPredictor : IDisposable
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer? _model;
        private readonly PredictionEngine<ImageInput, LandmarkOutput>? _predictionEngine;
        private readonly string _modelPath;
        private readonly bool _isModelLoaded;
        private bool _disposed = false;

        public OnnxLandmarkPredictor(string modelPath)
        {
            _modelPath = modelPath;
            _mlContext = new MLContext(seed: 42);

            try
            {
                // Check if model file exists and has correct extension
                if (File.Exists(modelPath) && Path.GetExtension(modelPath).ToLower() == ".onnx")
                {
                    // Create the pipeline for ONNX model
                    var pipeline = _mlContext.Transforms.ApplyOnnxModel(
                        modelFile: modelPath,
                        outputColumnNames: new[] { "output" },
                        inputColumnNames: new[] { "input" }
                    );

                    // Create a dummy input to fit the pipeline
                    var dummyData = new List<ImageInput> 
                    { 
                        new ImageInput { ImageData = new float[3 * 224 * 224] } 
                    };
                    var dummyDataView = _mlContext.Data.LoadFromEnumerable(dummyData);

                    // Fit the pipeline
                    _model = pipeline.Fit(dummyDataView);
                    _predictionEngine = _mlContext.Model.CreatePredictionEngine<ImageInput, LandmarkOutput>(_model);
                    _isModelLoaded = true;
                }
                else
                {
                    _isModelLoaded = false;
                }
            }
            catch (Exception)
            {
                _isModelLoaded = false;
                _model = null;
                _predictionEngine = null;
            }
        }

        /// <summary>
        /// Predict landmarks using the ONNX model
        /// </summary>
        public ShapeDetectionResult PredictLandmarks(Image<Rgb24> image, Rectangle rect)
        {
            if (!_isModelLoaded || _predictionEngine == null)
            {
                // Fallback to dummy landmarks if model not loaded
                return GenerateDummyLandmarks(rect);
            }

            try
            {
                // Preprocess image for model input
                var preprocessedImage = PreprocessImageForModel(image, rect);
                
                // Create input data
                var input = new ImageInput { ImageData = preprocessedImage };
                
                // Run prediction
                var prediction = _predictionEngine.Predict(input);
                
                // Convert output to landmark points
                return ConvertOutputToLandmarks(prediction, rect);
            }
            catch (Exception)
            {
                // Fallback to dummy landmarks if prediction fails
                return GenerateDummyLandmarks(rect);
            }
        }

        /// <summary>
        /// Preprocess image for ONNX model input
        /// </summary>
        private float[] PreprocessImageForModel(Image<Rgb24> image, Rectangle rect)
        {
            const int modelWidth = 224;  // Adjust based on your model
            const int modelHeight = 224;
            
            // Extract the region of interest
            using var croppedImage = image.Clone();
            croppedImage.Mutate(x => x.Crop(new SixLabors.ImageSharp.Rectangle(rect.Left, rect.Top, rect.Width, rect.Height)));
            
            // Resize to model input size
            croppedImage.Mutate(x => x.Resize(modelWidth, modelHeight));
            
            // Convert to float array (CHW format: Channels, Height, Width)
            var imageData = new float[3 * modelHeight * modelWidth];
            
            croppedImage.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < modelHeight; y++)
                {
                    var pixelRow = accessor.GetRowSpan(y);
                    for (int x = 0; x < modelWidth; x++)
                    {
                        var pixel = pixelRow[x];
                        
                        // Normalize pixel values to [0, 1] and arrange in CHW format
                        imageData[0 * modelHeight * modelWidth + y * modelWidth + x] = pixel.R / 255.0f; // Red channel
                        imageData[1 * modelHeight * modelWidth + y * modelWidth + x] = pixel.G / 255.0f; // Green channel
                        imageData[2 * modelHeight * modelWidth + y * modelWidth + x] = pixel.B / 255.0f; // Blue channel
                    }
                }
            });
            
            return imageData;
        }

        /// <summary>
        /// Convert ONNX model output to landmark points
        /// </summary>
        private ShapeDetectionResult ConvertOutputToLandmarks(LandmarkOutput output, Rectangle rect)
        {
            var landmarks = output.Landmarks;
            var numLandmarks = landmarks.Length / 2; // Assuming x, y pairs
            
            var points = new LandmarkPoint[numLandmarks];
            
            for (int i = 0; i < numLandmarks; i++)
            {
                // Convert normalized coordinates back to image coordinates
                var x = landmarks[i * 2] * rect.Width + rect.Left;
                var y = landmarks[i * 2 + 1] * rect.Height + rect.Top;
                
                points[i] = new LandmarkPoint(x, y);
            }
            
            return new ShapeDetectionResult(points);
        }

        /// <summary>
        /// Generate dummy landmarks as fallback
        /// </summary>
        private ShapeDetectionResult GenerateDummyLandmarks(Rectangle rect)
        {
            const int numLandmarks = 68; // Default number
            var points = new LandmarkPoint[numLandmarks];
            var random = new Random(42);
            
            for (int i = 0; i < numLandmarks; i++)
            {
                var x = rect.Left + random.NextSingle() * rect.Width;
                var y = rect.Top + random.NextSingle() * rect.Height;
                points[i] = new LandmarkPoint(x, y);
            }
            
            return new ShapeDetectionResult(points);
        }

        /// <summary>
        /// Check if the model is loaded and ready
        /// </summary>
        public bool IsModelLoaded => _isModelLoaded;

        /// <summary>
        /// Get information about the loaded model
        /// </summary>
        public string GetModelInfo()
        {
            if (_isModelLoaded)
            {
                return $"ONNX model loaded from: {_modelPath}";
            }
            else
            {
                return $"Model not loaded. Expected ONNX file at: {_modelPath}";
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _predictionEngine?.Dispose();
                _disposed = true;
            }
        }
    }
}
