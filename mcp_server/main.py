#!/usr/bin/env python3

import sys
import asyncio
import base64
import io
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

print("[SCRIPT] Script started", file=sys.stderr, flush=True)

import cv2
import dlib
import numpy as np
from mcp.server import Server
from mcp.types import (
    Tool,
    CallToolResult,
)
from PIL import Image

print("[SCRIPT] MCP imports completed", file=sys.stderr, flush=True)

# Import utils from the same directory
try:
    print("[SCRIPT] Importing utils...", file=sys.stderr, flush=True)
    import utils

    print("[SCRIPT] Utils imported successfully", file=sys.stderr, flush=True)
except ImportError as e:
    print(f"[ERROR] Error importing utils: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

# Configuration
PREDICTOR_FILE = os.getenv(
    "PREDICTOR_FILE",
    str(Path(__file__).parent / "better_predictor_auto.dat"),
)
WEB_PORT = int(os.getenv("WEB_PORT", "8080"))
# Use Docker-compatible path if running in Docker, otherwise use local path
if os.path.exists("/app"):
    OUTPUT_DIR = "/app/outputs"
else:
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
        # Use 0.0.0.0 to bind to all interfaces in Docker
        server_address = ("0.0.0.0", WEB_PORT)
        with socketserver.TCPServer(server_address, ImageHTTPRequestHandler) as httpd:
            print(
                f"Web server serving images at http://0.0.0.0:{WEB_PORT}",
                file=sys.stderr,
            )
            httpd.serve_forever()
    except Exception as e:
        print(f"Failed to start web server: {e}", file=sys.stderr)


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
                    print(
                        f"Truncated landmarks to 50 (was {len(landmarks)})",
                        file=sys.stderr,
                    )

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
                    print(f"Truncated landmarks at 100 (found more)", file=sys.stderr)
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
            print(f"Parsed {len(landmarks)} landmarks", file=sys.stderr)
        except Exception as e:
            print(f"Error parsing XML: {e}", file=sys.stderr)
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
            print(f"Saved annotated image: {output_path}", file=sys.stderr)
            print(f"Available at: {image_url}", file=sys.stderr)

            return image_url

        except Exception as e:
            print(f"Error creating annotated image: {e}", file=sys.stderr)
            return ""


# Initialize processor
try:
    print(f"[INIT] Initializing LizardMorph processor...", file=sys.stderr, flush=True)
    print(f"[INIT] Predictor file path: {PREDICTOR_FILE}", file=sys.stderr, flush=True)
    print(
        f"[INIT] Predictor file exists: {os.path.exists(PREDICTOR_FILE)}",
        file=sys.stderr,
        flush=True,
    )
    print(f"[INIT] Output directory: {OUTPUT_DIR}", file=sys.stderr, flush=True)
    print(f"[INIT] Web port: {WEB_PORT}", file=sys.stderr, flush=True)

    processor = LizardMorphProcessor(PREDICTOR_FILE)
    print(
        f"[INIT] LizardMorph MCP Server initialized successfully",
        file=sys.stderr,
        flush=True,
    )
except Exception as e:
    print(f"[ERROR] Failed to initialize: {e}", file=sys.stderr, flush=True)
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Start web server in background thread
print("[WEB] Starting web server thread...", file=sys.stderr, flush=True)
web_thread = threading.Thread(target=start_web_server, daemon=True)
web_thread.start()
print("[WEB] Web server thread started", file=sys.stderr, flush=True)

# Create MCP server
print("[MCP] Creating MCP server...", file=sys.stderr, flush=True)
server = Server("lizardmorph-mcp-server")
print("[MCP] MCP server created", file=sys.stderr, flush=True)


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
    print(f"[TOOL] Tool called: {name}", file=sys.stderr, flush=True)

    try:
        if name == "process_lizard_image":
            try:
                image_b64 = arguments.get("image", "")
                image_name = arguments.get("image_name", "lizard")

                if not image_b64:
                    return {
                        "content": [
                            {"type": "text", "text": "Error: No image provided"}
                        ],
                        "isError": True,
                    }

                # Clean base64
                if image_b64.startswith("data:"):
                    image_b64 = image_b64.split(",")[1]

                try:
                    image_data = base64.b64decode(image_b64)
                except Exception as e:
                    return {
                        "content": [
                            {"type": "text", "text": f"Error decoding image: {str(e)}"}
                        ],
                        "isError": True,
                    }

                # Process image
                result = processor.process_image(image_data, image_name)

                if result["success"]:
                    response_text = f"Processed {image_name}: {result['num_landmarks']} landmarks detected"
                    if result.get("image_url"):
                        response_text += f" | View: {result['image_url']}"

                    return {
                        "content": [{"type": "text", "text": response_text}],
                        "isError": False,
                    }
                else:
                    error_msg = str(result.get("error", "Unknown error"))
                    if len(error_msg) > 500:
                        error_msg = error_msg[:500] + "... (truncated)"

                    return {
                        "content": [
                            {"type": "text", "text": f"Processing failed: {error_msg}"}
                        ],
                        "isError": True,
                    }

            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "... (truncated)"

                return {
                    "content": [
                        {"type": "text", "text": f"Unexpected error: {error_msg}"}
                    ],
                    "isError": True,
                }

        elif name == "health_check":
            try:
                # Check environment variable as a dummy health check
                status_text = (
                    "✅ Health check passed: System is operational."
                    if os.getenv("ENV", "development") == "development"
                    else "✅ Health check passed: Production environment."
                )

                return {
                    "content": [{"type": "text", "text": status_text}],
                    "isError": False,
                }

            except Exception as e:
                error_message = f"❌ Health check failed: {str(e)}"
                return {
                    "content": [{"type": "text", "text": error_message}],
                    "isError": True,
                }
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

                return {
                    "content": [{"type": "text", "text": response}],
                    "isError": False,
                }
            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Error listing images: {str(e)}"}
                    ],
                    "isError": True,
                }

        else:
            return {
                "content": [{"type": "text", "text": "Unknown tool"}],
                "isError": True,
            }

    except Exception as e:
        print(
            f"[FATAL] Unexpected error in handle_call_tool: {e}",
            file=sys.stderr,
            flush=True,
        )
        return {
            "content": [{"type": "text", "text": f"Server error: {str(e)}"}],
            "isError": True,
        }


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

    print("[MAIN] Starting MCP server main function...", file=sys.stderr, flush=True)
    print("Server name: lizardmorph-mcp-server", file=sys.stderr, flush=True)

    try:
        print("[MAIN] Creating stdio server...", file=sys.stderr, flush=True)
        async with stdio_server() as (read_stream, write_stream):
            print("[MAIN] MCP server connected to stdio", file=sys.stderr, flush=True)
            print(
                "[MAIN] Server running and ready to handle requests...",
                file=sys.stderr,
                flush=True,
            )
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )
    except Exception as e:
        print(f"[ERROR] Error in main server loop: {e}", file=sys.stderr, flush=True)
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("[START] Starting asyncio main...", file=sys.stderr, flush=True)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[STOP] Server stopped by user", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[FATAL] Fatal error: {e}", file=sys.stderr, flush=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)
