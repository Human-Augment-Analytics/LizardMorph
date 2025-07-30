using System.Xml.Linq;
using System.Text;
using LizardMorph.MCP.ImageProcessing;

namespace LizardMorph.MCP.FileFormats
{
    /// <summary>
    /// TPS file operations for landmark data
    /// </summary>
    public static class TpsProcessor
    {
        /// <summary>
        /// Convert XML landmark data to TPS format
        /// </summary>
        public static void ConvertXmlToTps(string xmlFilePath, string tpsFilePath)
        {
            var document = XDocument.Load(xmlFilePath);
            var id = 0;

            using var writer = new StreamWriter(tpsFilePath, false, Encoding.UTF8);

            foreach (var imagesElement in document.Descendants("images"))
            {
                foreach (var imageElement in imagesElement.Elements("image"))
                {
                    foreach (var boxElement in imageElement.Elements("box"))
                    {
                        var parts = boxElement.Elements("part").ToList();
                        
                        // Write landmark count
                        writer.WriteLine($"LM={parts.Count}");

                        var boxHeight = float.Parse(boxElement.Attribute("height")?.Value ?? "0");

                        // Write coordinates (flip Y coordinate for TPS format)
                        foreach (var part in parts)
                        {
                            var x = float.Parse(part.Attribute("x")?.Value ?? "0");
                            var y = float.Parse(part.Attribute("y")?.Value ?? "0");
                            
                            // Flip Y coordinate: TPS uses bottom-left origin
                            var flippedY = boxHeight + 2 - y;
                            
                            writer.WriteLine($"{x} {flippedY}");
                        }

                        // Write image name and ID
                        var imageFile = imageElement.Attribute("file")?.Value ?? "";
                        var imageName = Path.GetFileNameWithoutExtension(imageFile);
                        writer.WriteLine($"IMAGE={imageName}");
                        writer.WriteLine($"ID={id}");
                        id++;
                    }
                }
            }
        }

        /// <summary>
        /// Generate TPS file directly from landmark coordinates
        /// </summary>
        public static void GenerateTpsFile(LandmarkPoint[] landmarks, string imageName, int imageHeight, string outputPath)
        {
            using var writer = new StreamWriter(outputPath, false, Encoding.UTF8);
            
            // Write landmark count
            writer.WriteLine($"LM={landmarks.Length}");

            // Write coordinates (flip Y coordinate for TPS format)
            foreach (var landmark in landmarks)
            {
                var flippedY = imageHeight - landmark.Y;
                writer.WriteLine($"{landmark.X} {flippedY}");
            }

            // Write image name
            var baseImageName = Path.GetFileNameWithoutExtension(imageName);
            writer.WriteLine($"IMAGE={baseImageName}");
        }
    }
}
