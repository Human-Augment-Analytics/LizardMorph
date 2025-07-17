import asyncio
import base64
import io
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

import cv2
import dlib
import numpy as np
from mcp.server import Server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)
from PIL import Image

# Import utils from the same directory
try:
    import utils
except ImportError as e:
    print(f"Error importing utils: {e}")
    sys.exit(1)

# Configuration
PREDICTOR_FILE = os.getenv(
    "PREDICTOR_FILE",
    str(Path(__file__).parent / "better_predictor_auto.dat"),
)
WEB_PORT = int(os.getenv("WEB_PORT", "8080"))
# Use local outputs directory for Windows
OUTPUT_DIR = str(Path(__file__).parent / "outputs")


class ImageHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler to serve images from the outputs directory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=OUTPUT_DIR, **kwargs)

    def end_headers(self):
        # Add CORS headers to allow cross-origin requests
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def start_web_server():
    """Start the HTTP server in a separate thread."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with socketserver.TCPServer(("", WEB_PORT), ImageHTTPRequestHandler) as httpd:
            print(f"Web server serving images at http://localhost:{WEB_PORT}")
            httpd.serve_forever()
    except Exception as e:
        print(f"Failed to start web server: {e}")


class LizardMorphProcessor:
    def __init__(self, predictor_file: str):
        self.predictor_file = predictor_file
        if not os.path.exists(predictor_file):
            raise FileNotFoundError(f"Predictor file not found: {predictor_file}")

    def process_image(
        self, image_data: bytes, image_name: str = "lizard"
    ) -> Dict[str, Any]:
        """Process an image and return landmark predictions with image URL."""
        try:
            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_file.write(image_data)
                temp_image_path = temp_file.name

            with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as temp_xml:
                temp_xml_path = temp_xml.name

            try:
                # Process the image
                utils.predictions_to_xml_single(
                    self.predictor_file, temp_image_path, temp_xml_path
                )

                # Parse landmarks (limit to prevent large responses)
                landmarks = self._parse_xml_landmarks(temp_xml_path)

                # Limit landmarks to first 50 to prevent response length issues
                if len(landmarks) > 50:
                    landmarks = landmarks[:50]
                    print(f"Truncated landmarks to 50 (was {len(landmarks)})")

                # Get image info
                image = Image.open(temp_image_path)
                width, height = image.size

                # Create annotated image and save it
                image_url = self._create_and_save_annotated_image(
                    temp_image_path, landmarks, image_name
                )

                return {
                    "success": True,
                    "landmarks": landmarks,
                    "image_width": width,
                    "image_height": height,
                    "num_landmarks": len(landmarks),
                    "image_url": image_url,
                }

            finally:
                try:
                    os.unlink(temp_image_path)
                    os.unlink(temp_xml_path)
                except:
                    pass

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def _parse_xml_landmarks(self, xml_path: str) -> List[Dict[str, float]]:
        """Parse XML file to extract landmark coordinates."""
        import xml.etree.ElementTree as ET

        landmarks = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for i, part in enumerate(root.findall(".//part")):
                # Limit to first 100 landmarks to prevent response length issues
                if i >= 100:
                    print(f"Truncated landmarks at 100 (found more)")
                    break

                name = part.get("name")
                x = float(part.get("x"))
                y = float(part.get("y"))
                landmarks.append(
                    {
                        "id": int(name) if name and name.isdigit() else i,
                        "x": round(x, 1),  # Round to reduce precision
                        "y": round(y, 1),  # Round to reduce precision
                        "name": f"landmark_{name}" if name else f"landmark_{i}",
                    }
                )
            landmarks.sort(key=lambda l: l["id"])
            print(f"Parsed {len(landmarks)} landmarks")
        except Exception as e:
            print(f"Error parsing XML: {e}")
        return landmarks

    def _create_and_save_annotated_image(
        self, image_path: str, landmarks: List[Dict[str, float]], image_name: str
    ) -> str:
        """Create annotated image with landmarks and save to web-accessible directory."""
        try:
            # Ensure output directory exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")

            # Convert BGR to RGB for consistent color handling
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize if too large (max 1200px on longest side for good quality)
            height, width = image_rgb.shape[:2]
            max_dimension = 1200

            if max(width, height) > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int((height * max_dimension) / width)
                else:
                    new_height = max_dimension
                    new_width = int((width * max_dimension) / height)

                image_rgb = cv2.resize(
                    image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA
                )

                # Scale landmarks accordingly
                scale_x = new_width / width
                scale_y = new_height / height
                scaled_landmarks = []
                for landmark in landmarks:
                    scaled_landmarks.append(
                        {
                            "id": landmark["id"],
                            "x": landmark["x"] * scale_x,
                            "y": landmark["y"] * scale_y,
                            "name": landmark["name"],
                        }
                    )
                landmarks = scaled_landmarks

            # Draw landmarks with bright, visible colors
            for landmark in landmarks:
                x, y = int(landmark["x"]), int(landmark["y"])

                # Draw larger, more visible landmarks
                cv2.circle(image_rgb, (x, y), 6, (255, 0, 0), -1)  # Red filled circle
                cv2.circle(image_rgb, (x, y), 8, (0, 255, 255), 2)  # Cyan outline

                # Add landmark ID with better visibility
                cv2.putText(
                    image_rgb,
                    str(landmark["id"]),
                    (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )  # White text with thickness
                cv2.putText(
                    image_rgb,
                    str(landmark["id"]),
                    (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )  # Black outline

            # Generate unique filename with timestamp
            timestamp = int(time.time())
            safe_name = "".join(
                c for c in image_name if c.isalnum() or c in ("-", "_")
            ).rstrip()
            output_filename = f"{safe_name}_landmarks_{timestamp}.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            # Convert back to BGR for OpenCV saving
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Save the annotated image with high quality
            cv2.imwrite(output_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Return the URL (assuming the server is accessible)
            image_url = f"http://localhost:{WEB_PORT}/{output_filename}"
            print(f"Saved annotated image: {output_path}")
            print(f"Available at: {image_url}")

            return image_url

        except Exception as e:
            print(f"Error creating annotated image: {e}")
            return ""


# Initialize processor
try:
    processor = LizardMorphProcessor(PREDICTOR_FILE)
    print(f"LizardMorph MCP Server initialized")
except Exception as e:
    print(f"Failed to initialize: {e}")
    sys.exit(1)

# Start web server in background thread
web_thread = threading.Thread(target=start_web_server, daemon=True)
web_thread.start()

# Create MCP server
server = Server("lizardmorph-mcp-server")


@server.list_tools()
async def handle_list_tools():
    """List available tools."""
    return [
        Tool(
            name="process_lizard_image",
            description="Process lizard X-ray images to detect anatomical landmarks and return image URL",
            inputSchema={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Base64 encoded image"},
                    "image_name": {
                        "type": "string",
                        "default": "lizard",
                        "description": "Name for the processed image",
                    },
                },
                "required": ["image"],
            },
        ),
        Tool(
            name="health_check",
            description="Check server status",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_processed_images",
            description="List all processed images available",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    """Handle tool calls."""

    if name == "process_lizard_image":
        try:
            image_b64 = arguments.get("image", "")
            image_name = arguments.get("image_name", "lizard")

            if not image_b64:
                error_text = TextContent(type="text", text="Error: No image provided")
                return CallToolResult(content=[error_text], isError=True)

            # Clean base64
            if image_b64.startswith("data:"):
                image_b64 = image_b64.split(",")[1]

            try:
                image_data = base64.b64decode(image_b64)
            except Exception as e:
                decode_error_text = TextContent(
                    type="text", text=f"Error decoding image: {str(e)}"
                )
                return CallToolResult(content=[decode_error_text], isError=True)

            # Process image
            result = processor.process_image(image_data, image_name)

            # Debug: Check response size
            print(
                f"Processing result keys: {list(result.keys()) if result else 'None'}"
            )
            if result.get("landmarks"):
                print(f"Number of landmarks: {len(result['landmarks'])}")
                # Don't print first landmark to avoid potential large data in logs
                print(
                    f"Landmark keys: {list(result['landmarks'][0].keys()) if result['landmarks'] else 'None'}"
                )

            if result["success"]:
                # Create ultra-minimal response to avoid length issues
                response_text = f"✅ Processed {image_name}: {result['num_landmarks']} landmarks detected"

                if result.get("image_url"):
                    response_text += f" | View: {result['image_url']}"

                # Debug: Check response length
                print(f"Response length: {len(response_text)} characters")
                print(f"Final response: {response_text}")

                text_content = TextContent(type="text", text=response_text)
                return CallToolResult(content=[text_content], isError=False)
            else:
                # Ensure error message is not too long
                error_msg = str(result.get("error", "Unknown error"))
                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "... (truncated)"

                error_text = TextContent(
                    type="text", text=f"❌ Processing failed: {error_msg}"
                )
                return CallToolResult(content=[error_text], isError=True)

        except Exception as e:
            # Ensure exception message is not too long
            error_msg = str(e)
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "... (truncated)"

            error_text = TextContent(type="text", text=f"Unexpected error: {error_msg}")
            return CallToolResult(content=[error_text], isError=True)

    elif name == "health_check":
        try:
            predictor_status = (
                "✅ Available" if os.path.exists(PREDICTOR_FILE) else "❌ Missing"
            )
            web_status = f"✅ Running on port {WEB_PORT}"

            status_text = f"Predictor: {predictor_status}, Web: {web_status}"

            status_text_content = TextContent(type="text", text=status_text)
            return CallToolResult(content=[status_text_content], isError=False)
        except Exception as e:
            health_error_text = TextContent(
                type="text", text=f"Health check error: {str(e)}"
            )
            return CallToolResult(content=[health_error_text], isError=True)

    elif name == "list_processed_images":
        try:
            if os.path.exists(OUTPUT_DIR):
                images = [
                    f
                    for f in os.listdir(OUTPUT_DIR)
                    if f.endswith((".jpg", ".jpeg", ".png"))
                ]
                if images:
                    image_list = "\n".join(
                        [
                            f"- {img} - http://localhost:{WEB_PORT}/{img}"
                            for img in sorted(images)
                        ]
                    )
                    response = f"# Processed Images\n\n{image_list}"
                else:
                    response = "No processed images found."
            else:
                response = "Output directory not found."

            response_text_content = TextContent(type="text", text=response)
            return CallToolResult(content=[response_text_content], isError=False)
        except Exception as e:
            list_error_text = TextContent(
                type="text", text=f"Error listing images: {str(e)}"
            )
            return CallToolResult(content=[list_error_text], isError=True)

    else:
        unknown_tool_text = TextContent(type="text", text="Unknown tool")
        return CallToolResult(content=[unknown_tool_text], isError=True)


@server.list_prompts()
async def handle_list_prompts():
    return []


@server.get_prompt()
async def handle_get_prompt(name: str, arguments: dict | None = None):
    raise ValueError(f"Unknown prompt: {name}")


@server.list_resources()
async def handle_list_resources():
    return []


@server.read_resource()
async def handle_read_resource(uri: str):
    raise ValueError(f"Unknown resource: {uri}")


async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
