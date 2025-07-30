using System.ComponentModel;
using System.Text;
using ModelContextProtocol.Server;
using LizardMorph.MCP.ImageProcessing;
using LizardMorph.MCP.FileFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LizardMorph.MCP.Tools
{
    /// <summary>
    /// LizardMorph MCP tools for processing lizard X-ray images and generating TPS files.
    /// These tools can be invoked by MCP clients to perform batch image processing operations.
    /// </summary>
    internal class LizardMorphTools
    {
        [McpServerTool]
        [Description("Process a folder of lizard X-ray images and generate TPS files with landmark predictions.")]
        public async Task<string> ProcessImagesFolder(
            [Description("Full path to the folder containing images to process")] string imagesFolder,
            [Description("Full path to the predictor .dat file (optional, defaults to ./better_predictor_auto.dat)")] string? predictorFile = null,
            [Description("Full path to the output directory where TPS files will be saved (optional, defaults to ./output)")] string? outputDirectory = null)
        {
            try
            {
                // Validate input parameters
                if (string.IsNullOrWhiteSpace(imagesFolder) || !Directory.Exists(imagesFolder))
                {
                    return $"Error: Images folder '{imagesFolder}' does not exist or is invalid.";
                }

                // Set defaults
                predictorFile ??= "./better_predictor_auto.dat";
                outputDirectory ??= "./output";

                // Create output directory if it doesn't exist
                Directory.CreateDirectory(outputDirectory);

                // Validate predictor file
                if (!File.Exists(predictorFile))
                {
                    return $"Error: Predictor file '{predictorFile}' not found.";
                }

                // Get all image files from the folder
                var imageExtensions = new[] { ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp" };
                var imageFiles = Directory.GetFiles(imagesFolder)
                    .Where(file => imageExtensions.Contains(Path.GetExtension(file).ToLower()))
                    .ToArray();

                if (imageFiles.Length == 0)
                {
                    return $"No image files found in folder '{imagesFolder}'. Supported formats: {string.Join(", ", imageExtensions)}";
                }

                var results = new List<string>();
                var processedCount = 0;
                var failedCount = 0;

                // Initialize the landmark predictor
                using var predictor = new LandmarkPredictor(predictorFile);

                foreach (var imagePath in imageFiles)
                {
                    try
                    {
                        var fileName = Path.GetFileNameWithoutExtension(imagePath);
                        var xmlOutputPath = Path.Combine(outputDirectory, $"output_{fileName}.xml");
                        var tpsOutputPath = Path.Combine(outputDirectory, $"output_{fileName}.tps");

                        // Process the image
                        await ProcessSingleImageAsync(imagePath, xmlOutputPath, tpsOutputPath, predictor);

                        processedCount++;
                        results.Add($"✓ Processed: {Path.GetFileName(imagePath)} -> {Path.GetFileName(tpsOutputPath)}");
                    }
                    catch (Exception ex)
                    {
                        failedCount++;
                        results.Add($"✗ Failed: {Path.GetFileName(imagePath)} - {ex.Message}");
                    }
                }

                var summary = new StringBuilder();
                summary.AppendLine($"Batch processing completed:");
                summary.AppendLine($"- Total images found: {imageFiles.Length}");
                summary.AppendLine($"- Successfully processed: {processedCount}");
                summary.AppendLine($"- Failed: {failedCount}");
                summary.AppendLine($"- Output directory: {outputDirectory}");
                summary.AppendLine();
                summary.AppendLine("Results:");
                foreach (var result in results)
                {
                    summary.AppendLine(result);
                }

                return summary.ToString();
            }
            catch (Exception ex)
            {
                return $"Error during batch processing: {ex.Message}";
            }
        }

        [McpServerTool]
        [Description("Check server status and verify .NET dependencies for LizardMorph image processing.")]
        public async Task<string> CheckStatus()
        {
            var status = new StringBuilder();
            status.AppendLine("LizardMorph MCP Server Status:");
            status.AppendLine();

            try
            {
                // Check .NET runtime
                var dotnetVersion = Environment.Version;
                status.AppendLine($"✓ .NET Runtime: {dotnetVersion}");

                // Check if we can load images
                try
                {
                    // Try to create a small test image
                    using var testImage = new Image<Rgb24>(10, 10);
                    status.AppendLine("✓ ImageSharp: Available");
                }
                catch (Exception ex)
                {
                    status.AppendLine($"✗ ImageSharp: Error - {ex.Message}");
                }

                // Check model availability
                var defaultPredictorPath = "./better_predictor_auto.dat";

                if (File.Exists(defaultPredictorPath))
                {
                    try
                    {
                        using var testPredictor = new LandmarkPredictor(defaultPredictorPath);
                        if (testPredictor.IsRealModelLoaded)
                        {
                            status.AppendLine("✓ Dlib Model: Loaded and ready");
                            status.AppendLine($"  {testPredictor.GetModelInfo()}");
                        }
                        else
                        {
                            status.AppendLine("✗ Dlib Model: .dat file found but failed to load");
                            status.AppendLine($"  Path: {defaultPredictorPath}");
                            status.AppendLine("  Error: Processing will fail without a valid model");
                        }
                    }
                    catch (Exception ex)
                    {
                        status.AppendLine($"✗ Dlib Model: Error loading - {ex.Message}");
                        status.AppendLine("  Error: Processing will fail without a valid model");
                    }
                }
                else
                {
                    status.AppendLine("✗ Predictor Model: No .dat file found");
                    status.AppendLine($"  Expected: {defaultPredictorPath}");
                    status.AppendLine("  Error: Processing will fail without a valid model");
                }

                // Check file system access
                try
                {
                    var tempPath = Path.GetTempPath();
                    var testFile = Path.Combine(tempPath, "lizardmorph_test.tmp");
                    await File.WriteAllTextAsync(testFile, "test");
                    File.Delete(testFile);
                    status.AppendLine("✓ File System: Read/Write access available");
                }
                catch (Exception ex)
                {
                    status.AppendLine($"✗ File System: Limited access - {ex.Message}");
                }

                // Check memory
                var workingSet = Environment.WorkingSet / (1024 * 1024);
                status.AppendLine($"✓ Memory: {workingSet} MB working set");

                status.AppendLine();
                status.AppendLine("DlibDotNet Integration:");
                status.AppendLine("✓ Using DlibDotNet for direct .dat file support");
                status.AppendLine("✓ No conversion needed - works directly with dlib models");
                status.AppendLine("⚠️ Requires valid .dat predictor model for landmark prediction");
                status.AppendLine();
                status.AppendLine("Server ready for image processing!");
            }
            catch (Exception ex)
            {
                status.AppendLine($"✗ Error checking status: {ex.Message}");
            }

            return status.ToString();
        }

        [McpServerTool]
        [Description("List all processed images and TPS files available in an output directory.")]
        public string ListProcessedImages(
            [Description("Path to the output directory to scan for processed files")] string outputDirectory = "./output")
        {
            try
            {
                if (!Directory.Exists(outputDirectory))
                {
                    return $"Output directory '{outputDirectory}' does not exist.";
                }

                var xmlFiles = Directory.GetFiles(outputDirectory, "*.xml");
                var tpsFiles = Directory.GetFiles(outputDirectory, "*.tps");
                var csvFiles = Directory.GetFiles(outputDirectory, "*.csv");

                var result = new StringBuilder();
                result.AppendLine($"Processed files in '{outputDirectory}':");
                result.AppendLine();

                result.AppendLine($"XML files ({xmlFiles.Length}):");
                foreach (var file in xmlFiles)
                {
                    var fileInfo = new FileInfo(file);
                    result.AppendLine($"  - {Path.GetFileName(file)} ({fileInfo.Length} bytes, {fileInfo.LastWriteTime:yyyy-MM-dd HH:mm})");
                }

                result.AppendLine();
                result.AppendLine($"TPS files ({tpsFiles.Length}):");
                foreach (var file in tpsFiles)
                {
                    var fileInfo = new FileInfo(file);
                    result.AppendLine($"  - {Path.GetFileName(file)} ({fileInfo.Length} bytes, {fileInfo.LastWriteTime:yyyy-MM-dd HH:mm})");
                }

                result.AppendLine();
                result.AppendLine($"CSV files ({csvFiles.Length}):");
                foreach (var file in csvFiles)
                {
                    var fileInfo = new FileInfo(file);
                    result.AppendLine($"  - {Path.GetFileName(file)} ({fileInfo.Length} bytes, {fileInfo.LastWriteTime:yyyy-MM-dd HH:mm})");
                }

                return result.ToString();
            }
            catch (Exception ex)
            {
                return $"Error listing processed images: {ex.Message}";
            }
        }

        [McpServerTool]
        [Description("Process a single lizard X-ray image and generate TPS file with landmark predictions.")]
        public async Task<string> ProcessSingleImage(
            [Description("Full path to the image file to process")] string imagePath,
            [Description("Full path to the predictor .dat file (optional, defaults to ./better_predictor_auto.dat)")] string? predictorFile = null,
            [Description("Full path to the output directory (optional, defaults to ./output)")] string? outputDirectory = null)
        {
            try
            {
                // Validate input
                if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
                {
                    return $"Error: Image file '{imagePath}' does not exist.";
                }

                // Set defaults
                predictorFile ??= "./better_predictor_auto.dat";
                outputDirectory ??= "./output";

                // Create output directory if it doesn't exist
                Directory.CreateDirectory(outputDirectory);

                // Validate predictor file
                if (!File.Exists(predictorFile))
                {
                    return $"Error: Predictor file '{predictorFile}' not found.";
                }

                var fileName = Path.GetFileNameWithoutExtension(imagePath);
                var xmlOutputPath = Path.Combine(outputDirectory, $"output_{fileName}.xml");
                var tpsOutputPath = Path.Combine(outputDirectory, $"output_{fileName}.tps");

                // Process the image
                using var predictor = new LandmarkPredictor(predictorFile);
                await ProcessSingleImageAsync(imagePath, xmlOutputPath, tpsOutputPath, predictor);

                return $"✓ Successfully processed: {Path.GetFileName(imagePath)}\n" +
                       $"  XML output: {xmlOutputPath}\n" +
                       $"  TPS output: {tpsOutputPath}";
            }
            catch (Exception ex)
            {
                return $"Error processing image: {ex.Message}";
            }
        }

        [McpServerTool]
        [Description("Process a single lizard X-ray image and generate an image with landmarks visualized as colored circles.")]
        public async Task<string> ProcessImageWithLandmarks(
            [Description("Full path to the image file to process")] string imagePath,
            [Description("Full path to the predictor .dat file (optional, defaults to ./better_predictor_auto.dat)")] string? predictorFile = null,
            [Description("Full path to the output directory (optional, defaults to ./output)")] string? outputDirectory = null,
            [Description("Radius of landmark points in pixels (optional, defaults to 3)")] int pointRadius = 3,
            [Description("Whether to number the landmarks (optional, defaults to false)")] bool numbered = false)
        {
            try
            {
                // Validate input
                if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
                {
                    return $"Error: Image file '{imagePath}' does not exist.";
                }

                // Set defaults
                predictorFile ??= "./better_predictor_auto.dat";
                outputDirectory ??= "./output";

                // Create output directory if it doesn't exist
                Directory.CreateDirectory(outputDirectory);

                // Validate predictor file
                if (!File.Exists(predictorFile))
                {
                    return $"Error: Predictor file '{predictorFile}' not found.";
                }

                var fileName = Path.GetFileNameWithoutExtension(imagePath);
                var landmarksImagePath = Path.Combine(outputDirectory, $"landmarks_{fileName}.jpg");
                var xmlOutputPath = Path.Combine(outputDirectory, $"output_{fileName}.xml");
                var tpsOutputPath = Path.Combine(outputDirectory, $"output_{fileName}.tps");

                // Process the image with landmarks visualization
                using var predictor = new LandmarkPredictor(predictorFile);
                await ProcessImageWithLandmarksAsync(imagePath, landmarksImagePath, xmlOutputPath, tpsOutputPath, predictor, pointRadius, numbered);

                return $"✓ Successfully processed with landmarks: {Path.GetFileName(imagePath)}\n" +
                       $"  Landmarks image: {landmarksImagePath}\n" +
                       $"  XML output: {xmlOutputPath}\n" +
                       $"  TPS output: {tpsOutputPath}";
            }
            catch (Exception ex)
            {
                return $"Error processing image with landmarks: {ex.Message}";
            }
        }

        [McpServerTool]
        [Description("Process a folder of lizard X-ray images and generate images with landmarks visualized as colored circles.")]
        public async Task<string> ProcessFolderWithLandmarks(
            [Description("Full path to the folder containing images to process")] string imagesFolder,
            [Description("Full path to the predictor .dat file (optional, defaults to ./better_predictor_auto.dat)")] string? predictorFile = null,
            [Description("Full path to the output directory where files will be saved (optional, defaults to ./output)")] string? outputDirectory = null,
            [Description("Radius of landmark points in pixels (optional, defaults to 50)")] int pointRadius = 50,
            [Description("Whether to number the landmarks (optional, defaults to false)")] bool numbered = false)
        {
            try
            {
                // Validate input parameters
                if (string.IsNullOrWhiteSpace(imagesFolder) || !Directory.Exists(imagesFolder))
                {
                    return $"Error: Images folder '{imagesFolder}' does not exist or is invalid.";
                }

                // Set defaults
                predictorFile ??= "./better_predictor_auto.dat";
                outputDirectory ??= "./output";

                // Create output directory if it doesn't exist
                Directory.CreateDirectory(outputDirectory);

                // Validate predictor file
                if (!File.Exists(predictorFile))
                {
                    return $"Error: Predictor file '{predictorFile}' not found.";
                }

                // Get all image files from the folder
                var imageExtensions = new[] { ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp" };
                var imageFiles = Directory.GetFiles(imagesFolder)
                    .Where(file => imageExtensions.Contains(Path.GetExtension(file).ToLower()))
                    .ToArray();

                if (imageFiles.Length == 0)
                {
                    return $"No image files found in folder '{imagesFolder}'. Supported formats: {string.Join(", ", imageExtensions)}";
                }

                var results = new List<string>();
                var processedCount = 0;
                var failedCount = 0;

                // Initialize the landmark predictor
                using var predictor = new LandmarkPredictor(predictorFile);

                foreach (var imagePath in imageFiles)
                {
                    try
                    {
                        var fileName = Path.GetFileNameWithoutExtension(imagePath);
                        var landmarksImagePath = Path.Combine(outputDirectory, $"landmarks_{fileName}.jpg");
                        var xmlOutputPath = Path.Combine(outputDirectory, $"output_{fileName}.xml");
                        var tpsOutputPath = Path.Combine(outputDirectory, $"output_{fileName}.tps");

                        // Process the image with landmarks
                        await ProcessImageWithLandmarksAsync(imagePath, landmarksImagePath, xmlOutputPath, tpsOutputPath, predictor, pointRadius, numbered);

                        processedCount++;
                        results.Add($"✓ Processed: {Path.GetFileName(imagePath)} -> {Path.GetFileName(landmarksImagePath)}");
                    }
                    catch (Exception ex)
                    {
                        failedCount++;
                        results.Add($"✗ Failed: {Path.GetFileName(imagePath)} - {ex.Message}");
                    }
                }

                var summary = new StringBuilder();
                summary.AppendLine($"Batch processing with landmarks completed:");
                summary.AppendLine($"- Total images found: {imageFiles.Length}");
                summary.AppendLine($"- Successfully processed: {processedCount}");
                summary.AppendLine($"- Failed: {failedCount}");
                summary.AppendLine($"- Output directory: {outputDirectory}");
                summary.AppendLine($"- Point radius: {pointRadius} pixels");
                summary.AppendLine($"- Numbered landmarks: {numbered}");
                summary.AppendLine();
                summary.AppendLine("Results:");
                foreach (var result in results)
                {
                    summary.AppendLine(result);
                }

                return summary.ToString();
            }
            catch (Exception ex)
            {
                return $"Error during batch processing with landmarks: {ex.Message}";
            }
        }

        /// <summary>
        /// Process a single image and generate XML and TPS files
        /// </summary>
        private static async Task ProcessSingleImageAsync(string imagePath, string xmlOutputPath, string tpsOutputPath, LandmarkPredictor predictor)
        {
            // Load and preprocess the image
            using var originalImage = ImageProcessor.LoadAndConvertImage(imagePath);
            using var processedImage = ImageProcessor.PreprocessImage(originalImage);

            // Define scales for multi-scale processing
            var scales = new float[] { 0.25f, 0.5f, 1.0f };

            // Perform multi-scale landmark prediction
            var landmarks = predictor.PredictMultiScale(processedImage, scales);

            // Generate XML file
            XmlProcessor.GenerateXmlForImage(imagePath, landmarks, originalImage.Width, originalImage.Height, xmlOutputPath);

            // Generate TPS file from XML
            TpsProcessor.ConvertXmlToTps(xmlOutputPath, tpsOutputPath);

            await Task.CompletedTask; // For async signature compatibility
        }

        /// <summary>
        /// Process a single image, generate XML and TPS files, and create an image with landmarks visualized
        /// </summary>
        private static async Task ProcessImageWithLandmarksAsync(string imagePath, string landmarksImagePath, string xmlOutputPath, string tpsOutputPath, LandmarkPredictor predictor, int pointRadius, bool numbered)
        {
            // Load and preprocess the image
            using var originalImage = ImageProcessor.LoadAndConvertImage(imagePath);
            using var processedImage = ImageProcessor.PreprocessImage(originalImage);

            // Define scales for multi-scale processing
            var scales = new float[] { 0.25f, 0.5f, 1.0f };

            // Perform multi-scale landmark prediction
            var landmarks = predictor.PredictMultiScale(processedImage, scales);

            // Generate XML file
            XmlProcessor.GenerateXmlForImage(imagePath, landmarks, originalImage.Width, originalImage.Height, xmlOutputPath);

            // Generate TPS file from XML
            TpsProcessor.ConvertXmlToTps(xmlOutputPath, tpsOutputPath);

            // Generate image with landmarks
            using var imageWithLandmarks = numbered
                ? ImageProcessor.DrawNumberedLandmarksOnImage(originalImage, landmarks, pointRadius)
                : ImageProcessor.DrawLandmarksOnImage(originalImage, landmarks, pointRadius);

            // Save the image with landmarks
            await imageWithLandmarks.SaveAsJpegAsync(landmarksImagePath);

            await Task.CompletedTask; // For async signature compatibility
        }
    }
}
