using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Numerics;

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
    }
}
