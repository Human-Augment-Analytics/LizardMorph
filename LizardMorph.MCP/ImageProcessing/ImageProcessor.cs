using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.Fonts;
using SixLabors.ImageSharp.Drawing;

namespace LizardMorph.MCP.ImageProcessing
{
    /// <summary>
    /// Image processing utilities for lizard X-ray images
    /// </summary>
    public static class ImageProcessor
    {
        /// <summary>
        /// Apply bilateral filter approximation using Gaussian blur
        /// </summary>
        public static void ApplyBilateralFilter(Image<Rgb24> image, float sigmaColor = 41f, float sigmaSpace = 21f)
        {
            // Approximate bilateral filter with Gaussian blur
            // In practice, bilateral filtering is complex to implement efficiently
            // For MVP, we'll use a series of Gaussian blurs which provides similar smoothing
            image.Mutate(x => x.GaussianBlur(sigmaSpace / 10f));
        }

        /// <summary>
        /// Apply a 7x7 averaging kernel filter
        /// </summary>
        public static void ApplyAverageFilter(Image<Rgb24> image)
        {
            // Apply a box blur which approximates the averaging kernel
            image.Mutate(x => x.BoxBlur(3)); // 7x7 kernel approximated with radius 3
        }

        /// <summary>
        /// Resize image by scale factor
        /// </summary>
        public static Image<Rgb24> ResizeImage(Image<Rgb24> image, float scale)
        {
            var newWidth = (int)(image.Width * scale);
            var newHeight = (int)(image.Height * scale);

            var resized = image.Clone();
            resized.Mutate(x => x.Resize(newWidth, newHeight));
            return resized;
        }

        /// <summary>
        /// Convert image to RGB24 format if needed
        /// </summary>
        public static Image<Rgb24> LoadAndConvertImage(string imagePath)
        {
            using var originalImage = Image.Load(imagePath);
            return originalImage.CloneAs<Rgb24>();
        }

        /// <summary>
        /// Apply preprocessing pipeline to image
        /// </summary>
        public static Image<Rgb24> PreprocessImage(Image<Rgb24> image)
        {
            var processed = image.Clone();

            // Apply averaging filter (equivalent to cv2.filter2D with averaging kernel)
            ApplyAverageFilter(processed);

            // Apply bilateral filter approximation
            ApplyBilateralFilter(processed, 41f, 21f);

            return processed;
        }

        /// <summary>
        /// Create an image with landmarks overlaid as colored circles
        /// </summary>
        public static Image<Rgb24> DrawLandmarksOnImage(Image<Rgb24> originalImage, LandmarkPoint[] landmarks, int pointRadius = 3)
        {
            var imageWithLandmarks = originalImage.Clone();

            imageWithLandmarks.Mutate(x =>
            {
                // Define colors for different landmarks (cycling through colors)
                var colors = new[]
                {
                    Color.Red,
                    Color.Blue,
                    Color.Green,
                    Color.Yellow,
                    Color.Orange,
                    Color.Purple,
                    Color.Cyan,
                    Color.Magenta,
                    Color.Lime,
                    Color.Pink
                };

                for (int i = 0; i < landmarks.Length; i++)
                {
                    var landmark = landmarks[i];
                    var color = colors[i % colors.Length];

                    // Draw filled circle for the landmark using a simple approach
                    var ellipse = new EllipsePolygon(new PointF(landmark.X, landmark.Y), pointRadius);
                    x.Fill(color, ellipse);

                    // Draw a smaller white center for better visibility
                    if (pointRadius > 2)
                    {
                        var innerEllipse = new EllipsePolygon(new PointF(landmark.X, landmark.Y), pointRadius - 1);
                        x.Fill(Color.White, innerEllipse);
                    }
                }
            });

            return imageWithLandmarks;
        }

        /// <summary>
        /// Create an image with landmarks overlaid and numbered
        /// </summary>
        public static Image<Rgb24> DrawNumberedLandmarksOnImage(Image<Rgb24> originalImage, LandmarkPoint[] landmarks, int pointRadius = 5)
        {
            var imageWithLandmarks = originalImage.Clone();

            imageWithLandmarks.Mutate(x =>
            {
                // Use a simple approach for font
                Font font;
                try
                {
                    font = SystemFonts.CreateFont("Arial", 12, FontStyle.Bold);
                }
                catch
                {
                    // Use default system font as fallback
                    font = SystemFonts.CreateFont(SystemFonts.Families.First().Name, 12);
                }

                for (int i = 0; i < landmarks.Length; i++)
                {
                    var landmark = landmarks[i];

                    // Draw filled circle for the landmark
                    var ellipse = new EllipsePolygon(new PointF(landmark.X, landmark.Y), pointRadius);
                    x.Fill(Color.Red, ellipse);

                    // Draw white border around the circle
                    x.Draw(Color.White, 2, ellipse);

                    // Draw the landmark number
                    var text = (i + 1).ToString();
                    var textPosition = new PointF(landmark.X + pointRadius + 2, landmark.Y - 6);

                    // Draw text with white color for visibility
                    x.DrawText(text, font, Color.White, textPosition);
                }
            });

            return imageWithLandmarks;
        }
    }
}
