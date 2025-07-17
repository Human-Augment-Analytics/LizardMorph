import asyncio
import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import dlib
import numpy as np
from mcp.server import Server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    Role,
    ReadResourceResult,
    Resource,
)
from PIL import Image

# Import utils from the same directory
try:
    import utils
except ImportError as e:
    print(f"Error importing utils: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

# Configuration
PREDICTOR_FILE = os.getenv(
    "PREDICTOR_FILE",
    str(Path(__file__).parent / "better_predictor_auto.dat"),
)
TEMP_DIR = tempfile.gettempdir()


class LizardMorphProcessor:
    def __init__(self, predictor_file: str):
        self.predictor_file = predictor_file
        if not os.path.exists(predictor_file):
            raise FileNotFoundError(f"Predictor file not found: {predictor_file}")

    def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Process an image and return landmark predictions.

        Args:
            image_data: Raw image data bytes

        Returns:
            Dictionary containing landmarks and metadata
        """
        try:
            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_file.write(image_data)
                temp_image_path = temp_file.name

            # Create temporary XML output file
            with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as temp_xml:
                temp_xml_path = temp_xml.name

            try:
                # Use the existing utils function to process the image
                utils.predictions_to_xml_single(
                    self.predictor_file, temp_image_path, temp_xml_path
                )

                # Parse the XML to extract landmarks
                landmarks = self._parse_xml_landmarks(temp_xml_path)

                # Get image dimensions
                image = Image.open(temp_image_path)
                width, height = image.size

                return {
                    "success": True,
                    "landmarks": landmarks,
                    "image_width": width,
                    "image_height": height,
                    "num_landmarks": len(landmarks),
                    "message": "Image processed successfully",
                }

            finally:
                # Clean up temporary files
                try:
                    os.unlink(temp_image_path)
                    os.unlink(temp_xml_path)
                except:
                    pass

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to process image",
            }

    def _parse_xml_landmarks(self, xml_path: str) -> List[Dict[str, float]]:
        """Parse XML file to extract landmark coordinates."""
        import xml.etree.ElementTree as ET

        landmarks = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Find all part elements
            for part in root.findall(".//part"):
                name = part.get("name")
                x = float(part.get("x"))
                y = float(part.get("y"))

                landmarks.append(
                    {
                        "id": int(name) if name else len(landmarks),
                        "x": x,
                        "y": y,
                        "name": (
                            f"landmark_{name}" if name else f"landmark_{len(landmarks)}"
                        ),
                    }
                )

            # Sort by ID to maintain consistent ordering
            landmarks.sort(key=lambda l: l["id"])

        except Exception as e:
            print(f"Error parsing XML landmarks: {e}")

        return landmarks


# Initialize processor
try:
    processor = LizardMorphProcessor(PREDICTOR_FILE)
    print(f"LizardMorph MCP Server initialized with predictor: {PREDICTOR_FILE}")
except Exception as e:
    print(f"Failed to initialize processor: {e}")
    sys.exit(1)


# Create the MCP server
server = Server("lizardmorph-mcp-server")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="process_lizard_image",
            description="Process lizard X-ray images to detect anatomical landmarks using machine learning",
            inputSchema={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Base64 encoded image data (JPEG, PNG, etc.)",
                    },
                    "image_name": {
                        "type": "string",
                        "description": "Optional name for the image",
                        "default": "uploaded_image",
                    },
                },
                "required": ["image"],
            },
        ),
        Tool(
            name="get_predictor_info",
            description="Get information about the current predictor model",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="health_check",
            description="Check if the MCP server is running properly",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> CallToolResult:
    """Handle tool calls."""

    if name == "process_lizard_image":
        try:
            # Get image data from arguments
            image_b64 = arguments.get("image")
            image_name = arguments.get("image_name", "uploaded_image")

            if not image_b64:
                return CallToolResult(
                    content=[
                        TextContent(type="text", text="Error: No image data provided")
                    ],
                    isError=True,
                )

            # Decode base64 image
            try:
                # Remove data URL prefix if present
                if image_b64.startswith("data:"):
                    image_b64 = image_b64.split(",")[1]

                image_data = base64.b64decode(image_b64)
            except Exception as e:
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=f"Error decoding image: {str(e)}")
                    ],
                    isError=True,
                )

            # Process the image
            result = processor.process_image(image_data)

            if result["success"]:
                # Format the response
                response_text = f"""# Lizard Image Analysis Results

**Image**: {image_name}
**Dimensions**: {result['image_width']} x {result['image_height']} pixels
**Landmarks Detected**: {result['num_landmarks']}

## Landmark Coordinates:
"""

                for landmark in result["landmarks"]:
                    response_text += f"- **{landmark['name']}** (ID: {landmark['id']}): x={landmark['x']:.1f}, y={landmark['y']:.1f}\n"

                response_text += f"\n## Processing Status: ✅ {result['message']}"

                return CallToolResult(
                    content=[TextContent(type="text", text=response_text)]
                )
            else:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text=f"❌ Processing failed: {result['error']}"
                        )
                    ],
                    isError=True,
                )

        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unexpected error: {str(e)}")],
                isError=True,
            )

    elif name == "get_predictor_info":
        try:
            predictor_exists = os.path.exists(PREDICTOR_FILE)
            predictor_size = os.path.getsize(PREDICTOR_FILE) if predictor_exists else 0

            info_text = f"""# Predictor Model Information

**Model File**: {PREDICTOR_FILE}
**Status**: {'✅ Available' if predictor_exists else '❌ Not Found'}
**Size**: {predictor_size / 1024 / 1024:.1f} MB
**Type**: dlib shape predictor (.dat file)

## Description:
This model is trained to detect anatomical landmarks on lizard X-ray images. It uses dlib's shape prediction algorithm to identify key anatomical points for morphometric analysis.
"""

            return CallToolResult(content=[TextContent(type="text", text=info_text)])

        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Error getting predictor info: {str(e)}"
                    )
                ],
                isError=True,
            )

    elif name == "health_check":
        try:
            # Check if predictor file exists
            predictor_status = (
                "✅ Available" if os.path.exists(PREDICTOR_FILE) else "❌ Missing"
            )

            # Check if required modules are available
            modules_status = []
            required_modules = ["cv2", "dlib", "numpy", "PIL"]

            for module in required_modules:
                try:
                    __import__(module)
                    modules_status.append(f"✅ {module}")
                except ImportError:
                    modules_status.append(f"❌ {module}")

            health_text = f"""# LizardMorph MCP Server Health Check

## Core Components:
- **Predictor Model**: {predictor_status}
- **Utils Module**: ✅ Available

## Dependencies:
{chr(10).join(modules_status)}

## Server Status: 🟢 Running
"""

            return CallToolResult(content=[TextContent(type="text", text=health_text)])

        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Health check failed: {str(e)}")
                ],
                isError=True,
            )

    else:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")],
            isError=True,
        )


@server.list_prompts()
async def handle_list_prompts() -> list[Prompt]:
    """List available prompts."""
    return []


@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict | None = None
) -> GetPromptResult:
    """Get a prompt."""
    raise ValueError(f"Unknown prompt: {name}")


@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List available resources."""
    return []


@server.read_resource()
async def handle_read_resource(uri: str) -> ReadResourceResult:
    """Read a resource."""
    raise ValueError(f"Unknown resource: {uri}")


async def main():
    """Run the MCP server."""
    # Use stdin/stdout for MCP communication
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
