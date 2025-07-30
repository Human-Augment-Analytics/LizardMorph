using System.Numerics;

namespace LizardMorph.MCP.ImageProcessing
{
    /// <summary>
    /// Represents a landmark point with x, y coordinates
    /// </summary>
    public struct LandmarkPoint
    {
        public float X { get; set; }
        public float Y { get; set; }

        public LandmarkPoint(float x, float y)
        {
            X = x;
            Y = y;
        }

        public Vector2 ToVector2() => new Vector2(X, Y);
    }

    /// <summary>
    /// Represents a rectangular region
    /// </summary>
    public struct Rectangle
    {
        public int Left { get; set; }
        public int Top { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public int Right => Left + Width;
        public int Bottom => Top + Height;

        public Rectangle(int left, int top, int width, int height)
        {
            Left = left;
            Top = top;
            Width = width;
            Height = height;
        }
    }

    /// <summary>
    /// Shape detection result containing multiple landmark points
    /// </summary>
    public class ShapeDetectionResult
    {
        public LandmarkPoint[] Points { get; set; } = Array.Empty<LandmarkPoint>();
        public int NumParts => Points.Length;

        public ShapeDetectionResult(LandmarkPoint[] points)
        {
            Points = points;
        }

        public ShapeDetectionResult(int numPoints)
        {
            Points = new LandmarkPoint[numPoints];
        }
    }
}
