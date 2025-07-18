# LizardMorph MCP Server

A Model Context Protocol (MCP) server for processing lizard X-ray images and detecting anatomical landmarks using machine learning.

## Features

- **Image Processing**: Process lizard X-ray images to detect anatomical landmarks
- **Machine Learning**: Uses trained dlib shape predictor models
- **MCP Integration**: Seamless integration with MCP-compatible clients
- **Base64 Support**: Accepts images as base64-encoded data
- **Landmark Detection**: Returns precise x,y coordinates for anatomical landmarks

## Tools Available

### 1. `process_lizard_image`
Process lizard X-ray images to detect anatomical landmarks.

**Parameters:**
- `image` (required): Base64 encoded image data (JPEG, PNG, etc.)
- `image_name` (optional): Name for the image (default: "uploaded_image")

**Returns:**
- Landmark coordinates (x, y) for each detected anatomical point
- Image dimensions
- Processing status and metadata

### 2. `get_predictor_info`
Get information about the current predictor model.

**Returns:**
- Model file path and status
- Model size and type
- Description of the model's capabilities

### 3. `health_check`
Check if the MCP server is running properly.

**Returns:**
- Server status
- Component availability
- Dependency status

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -e .
   ```

2. **Set Environment Variables:**
   ```bash
   # Optional: Set custom predictor file path
   export PREDICTOR_FILE="/path/to/your/predictor.dat"
   ```

3. **Run the Server:**
   ```bash
   python main.py
   ```

## Usage Example

The MCP server accepts images as base64-encoded strings and returns landmark coordinates:

```json
{
  "tool": "process_lizard_image",
  "arguments": {
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD...",
    "image_name": "lizard_xray_001.jpg"
  }
}
```

Response:
```json
{
  "landmarks": [
    {"id": 0, "x": 245.3, "y": 156.7, "name": "landmark_0"},
    {"id": 1, "x": 312.1, "y": 189.4, "name": "landmark_1"},
    ...
  ],
  "image_width": 640,
  "image_height": 480,
  "num_landmarks": 25
}
```

## Configuration

The server uses the following configuration:

- **Predictor File**: `../backend/better_predictor_auto.dat` (configurable via `PREDICTOR_FILE` env var)
- **Temporary Directory**: System temp directory for processing
- **Supported Formats**: JPEG, PNG, BMP, TIFF

## Integration

This MCP server integrates with the main LizardMorph application by:

1. Using the same `utils.py` functions for image processing
2. Leveraging the trained predictor model from the backend
3. Providing the same landmark detection capabilities via MCP protocol

## Error Handling

The server includes comprehensive error handling for:
- Invalid image data
- Missing predictor files
- Processing failures
- Network/IO errors

## Requirements

- Python 3.10+
- OpenCV
- dlib
- NumPy
- Pillow
- MCP library
- aiofiles

## License

Part of the LizardMorph project.