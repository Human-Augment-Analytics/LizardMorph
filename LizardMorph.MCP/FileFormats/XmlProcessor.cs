using System.Xml;
using System.Xml.Linq;
using System.Text;
using LizardMorph.MCP.ImageProcessing;

namespace LizardMorph.MCP.FileFormats
{
    /// <summary>
    /// XML file operations for landmark data
    /// </summary>
    public static class XmlProcessor
    {
        /// <summary>
        /// Initialize a new XML document for landmark data
        /// </summary>
        public static (XDocument root, XElement imagesElement) InitializeXml()
        {
            var root = new XDocument(
                new XElement("dataset",
                    new XElement("name"),
                    new XElement("comment"),
                    new XElement("images")
                )
            );

            var imagesElement = root.Root!.Element("images")!;
            return (root, imagesElement);
        }

        /// <summary>
        /// Create a box element for the image
        /// </summary>
        public static XElement CreateBox(int width, int height)
        {
            return new XElement("box",
                new XAttribute("top", "1"),
                new XAttribute("left", "1"),
                new XAttribute("width", (width - 2).ToString()),
                new XAttribute("height", (height - 2).ToString())
            );
        }

        /// <summary>
        /// Create a part element for a landmark point
        /// </summary>
        public static XElement CreatePart(float x, float y, int id)
        {
            return new XElement("part",
                new XAttribute("name", id.ToString()),
                new XAttribute("x", ((int)x).ToString()),
                new XAttribute("y", ((int)y).ToString())
            );
        }

        /// <summary>
        /// Save XML document with pretty formatting
        /// </summary>
        public static void SavePrettyXml(XDocument document, string outputPath)
        {
            var settings = new XmlWriterSettings
            {
                Indent = true,
                IndentChars = "   ",
                NewLineChars = "\n",
                Encoding = Encoding.UTF8
            };

            using var writer = XmlWriter.Create(outputPath, settings);
            document.Save(writer);
        }

        /// <summary>
        /// Generate XML file for a single image with landmarks
        /// </summary>
        public static void GenerateXmlForImage(string imagePath, LandmarkPoint[] landmarks, int imageWidth, int imageHeight, string outputPath)
        {
            var (document, imagesElement) = InitializeXml();

            var imageElement = new XElement("image",
                new XAttribute("file", imagePath)
            );

            var boxElement = CreateBox(imageWidth, imageHeight);

            // Sort landmarks by ID and add as parts
            for (int i = 0; i < landmarks.Length; i++)
            {
                var part = CreatePart(landmarks[i].X, landmarks[i].Y, i);
                boxElement.Add(part);
            }

            imageElement.Add(boxElement);
            imagesElement.Add(imageElement);

            SavePrettyXml(document, outputPath);
        }
    }
}
