# LizardMorph MCP Server

A Model Context Protocol (MCP) server for processing lizard X-ray images and generating TPS (Thin Plate Spline) files for morphological analysis. This server uses **DlibDotNet** for direct .dat file support and provides both batch and single image processing capabilities.

## Features

- **Native .NET Implementation**: Built with C# and DlibDotNet for optimal performance
- **Direct Dlib Model Support**: Works directly with .dat predictor files without Python dependencies
- **Batch Image Processing**: Process entire folders of lizard X-ray images
- **Single Image Processing**: Process individual images
- **TPS File Generation**: Generate TPS files with landmark predictions for morphological analysis
- **Status Checking**: Verify .NET dependencies and server health
- **File Management**: List and manage processed files
- **Model Validation**: Requires valid .dat predictor model files for processing

## Tools Available

### 1. `ProcessImagesFolder`
Process a folder of lizard X-ray images and generate TPS files with landmark predictions.

**Parameters:**
- `imagesFolder` (required): Full path to the folder containing images to process
- `predictorFile` (optional): Full path to the predictor .dat file (defaults to ./better_predictor_auto.dat)
- `outputDirectory` (optional): Full path to the output directory (defaults to ./output)

### 2. `ProcessSingleImage`
Process a single lizard X-ray image and generate TPS file with landmark predictions.

**Parameters:**
- `imagePath` (required): Full path to the image file to process
- `predictorFile` (optional): Full path to the predictor .dat file (defaults to ./better_predictor_auto.dat)
- `outputDirectory` (optional): Full path to the output directory (defaults to ./output)

### 3. `CheckStatus`
Check server status and verify .NET dependencies for LizardMorph image processing.

**Parameters:** None

### 4. `ListProcessedImages`
List all processed images and TPS files available in an output directory.

**Parameters:**
- `outputDirectory` (optional): Path to the output directory to scan (defaults to ./output)

## Prerequisites

### .NET Dependencies
The server requires .NET 8.0 runtime and the following NuGet packages (automatically managed):
- `DlibDotNet`: For landmark detection using native dlib models
- `SixLabors.ImageSharp`: For image processing
- `MathNet.Numerics`: For statistical operations

### Predictor Model File
You need a trained dlib predictor file (`.dat` format) for landmark detection. The default expected location is `./better_predictor_auto.dat` relative to where the server is run.

**Note**: A valid predictor model file is required for processing. The server will return an error if no valid model is available.

## Installation and Setup

### From NuGet (when published)
```bash
dotnet tool install --global LizardMorph.MCP
```

### Local Development
1. Clone the repository
2. Navigate to the LizardMorph.MCP directory
3. Build the project: `dotnet build`
4. Run locally: `dotnet run`

## Usage Examples

### Processing a Folder of Images
```json
{
  "method": "tools/call",
  "params": {
    "name": "ProcessImagesFolder",
    "arguments": {
      "imagesFolder": "/path/to/your/images",
      "predictorFile": "/path/to/better_predictor_auto.dat",
      "outputDirectory": "/path/to/output"
    }
  }
}
```

### Processing a Single Image
```json
{
  "method": "tools/call",
  "params": {
    "name": "ProcessSingleImage",
    "arguments": {
      "imagePath": "/path/to/your/image.jpg",
      "predictorFile": "/path/to/better_predictor_auto.dat",
      "outputDirectory": "/path/to/output"
    }
  }
}
```

### Checking Server Status
```json
{
  "method": "tools/call",
  "params": {
    "name": "CheckStatus",
    "arguments": {}
  }
}
```

### Listing Processed Files
```json
{
  "method": "tools/call",
  "params": {
    "name": "ListProcessedImages",
    "arguments": {
      "outputDirectory": "/path/to/output"
    }
  }
}
```

## Output Files

For each processed image, the server generates:
- **XML file**: Contains detailed landmark coordinates in structured format
- **TPS file**: Thin Plate Spline format for morphological analysis software

## Architecture

The server is built with a native .NET implementation that provides:
- **Multi-scale Processing**: Uses multiple image scales and computes median landmarks for improved accuracy
- **DlibDotNet Integration**: Direct integration with dlib models without external dependencies
- **ImageSharp Processing**: Advanced image preprocessing including bilateral filtering approximation
- **Strict Model Validation**: Ensures valid predictor models are loaded before processing

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tif, .tiff)
- BMP (.bmp)

## Error Handling

The server provides detailed error messages for common issues:
- Missing .NET dependencies
- Invalid file paths
- Unsupported image formats
- Processing failures
- Missing or invalid predictor model files

## Contributing

This MCP server is part of the larger LizardMorph project. For contributions and issues, please refer to the main project repository.

## Checklist before publishing to NuGet.org

- Test the MCP server locally using the steps below.
- Update the package metadata in the .csproj file, in particular the `<PackageId>`.
- Update `.mcp/server.json` to declare your MCP server's inputs.
  - See [configuring inputs](https://aka.ms/nuget/mcp/guide/configuring-inputs) for more details.
- Pack the project using `dotnet pack`.

The `bin/Release` directory will contain the package file (.nupkg), which can be [published to NuGet.org](https://learn.microsoft.com/nuget/nuget-org/publish-a-package).

## Developing locally

To test this MCP server from source code (locally) without using a built MCP server package, you can configure your IDE to run the project directly using `dotnet run`.

```json
{
  "servers": {
    "LizardMorph.MCP": {
      "type": "stdio",
      "command": "dotnet",
      "args": [
        "run",
        "--project",
        "<PATH TO PROJECT DIRECTORY>"
      ]
    }
  }
}
```

## Testing the MCP Server

Once configured, you can test the server by:

1. **Check Status**: Ask Copilot to check the server status to verify it's working correctly
2. **Process Images**: Use the ProcessSingleImage or ProcessImagesFolder tools to test image processing
3. **List Files**: Use ListProcessedImages to see generated output files

Example: "Process the image in my sample_image folder using the LizardMorph MCP server"

## Publishing to NuGet.org

1. Run `dotnet pack -c Release` to create the NuGet package
2. Publish to NuGet.org with `dotnet nuget push bin/Release/*.nupkg --api-key <your-api-key> --source https://api.nuget.org/v3/index.json`

## Using the MCP Server from NuGet.org

Once the MCP server package is published to NuGet.org, you can configure it in your preferred IDE. Both VS Code and Visual Studio use the `dnx` command to download and install the MCP server package from NuGet.org.

- **VS Code**: Create a `<WORKSPACE DIRECTORY>/.vscode/mcp.json` file
- **Visual Studio**: Create a `<SOLUTION DIRECTORY>\.mcp.json` file

For both VS Code and Visual Studio, the configuration file uses the following server definition:

```json
{
  "servers": {
    "LizardMorph.MCP": {
      "type": "stdio",
      "command": "dnx",
      "args": [
        "<your package ID here>",
        "--version",
        "<your package version here>",
        "--yes"
      ]
    }
  }
}
```

## More information

.NET MCP servers use the [ModelContextProtocol](https://www.nuget.org/packages/ModelContextProtocol) C# SDK. For more information about MCP:

- [Official Documentation](https://modelcontextprotocol.io/)
- [Protocol Specification](https://spec.modelcontextprotocol.io/)
- [GitHub Organization](https://github.com/modelcontextprotocol)

Refer to the VS Code or Visual Studio documentation for more information on configuring and using MCP servers:

- [Use MCP servers in VS Code (Preview)](https://code.visualstudio.com/docs/copilot/chat/mcp-servers)
- [Use MCP servers in Visual Studio (Preview)](https://learn.microsoft.com/visualstudio/ide/mcp-servers)
