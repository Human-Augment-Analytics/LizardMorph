#!/usr/bin/env python3
"""
Example usage of the LizardMorph MCP Server
This demonstrates how to send requests to the running MCP server.
"""

import base64
import json
from pathlib import Path


def create_image_processing_request(image_path: str, image_name: str = None) -> dict:
    """Create a request to process an image."""

    # Read and encode the image
    with open(image_path, "rb") as f:
        image_data = f.read()

    image_b64 = base64.b64encode(image_data).decode("utf-8")

    if not image_name:
        image_name = Path(image_path).name

    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "process_lizard_image",
            "arguments": {
                "image": f"data:image/jpeg;base64,{image_b64}",
                "image_name": image_name,
            },
        },
    }


def create_health_check_request() -> dict:
    """Create a health check request."""
    return {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "health_check", "arguments": {}},
    }


def create_predictor_info_request() -> dict:
    """Create a predictor info request."""
    return {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "get_predictor_info", "arguments": {}},
    }


def main():
    """Generate example requests for the MCP server."""

    print("🔧 LizardMorph MCP Server - Usage Examples")
    print("=" * 50)

    # Example 1: Health Check
    print("\n1. Health Check Request:")
    health_request = create_health_check_request()
    print(json.dumps(health_request, indent=2))

    # Example 2: Predictor Info
    print("\n2. Predictor Info Request:")
    predictor_request = create_predictor_info_request()
    print(json.dumps(predictor_request, indent=2))

    # Example 3: Image Processing (if sample image exists)
    sample_image = Path("../sample_image/0003_dorsal.jpg")
    if sample_image.exists():
        print("\n3. Image Processing Request:")
        print("   (Image data truncated for display)")

        # Show the structure without the full base64 data
        request_structure = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "process_lizard_image",
                "arguments": {
                    "image": "data:image/jpeg;base64,[BASE64_IMAGE_DATA]",
                    "image_name": "0003_dorsal.jpg",
                },
            },
        }
        print(json.dumps(request_structure, indent=2))

        # Show actual file size
        image_size = sample_image.stat().st_size
        print(f"\n   Actual image size: {image_size:,} bytes")

    else:
        print(f"\n3. ⚠️  Sample image not found at: {sample_image}")
        print("   Place a lizard X-ray image there to test image processing.")

    print("\n" + "=" * 50)
    print("📡 Server Status:")
    print("   The MCP server is running and ready to accept requests.")
    print("   Connect via MCP client protocol (stdio) to send these requests.")
    print("=" * 50)

    # Usage instructions
    print("\n📝 How to use:")
    print("1. The server is running via stdio (standard input/output)")
    print("2. Send JSON-RPC requests as shown above")
    print("3. The server will respond with landmark coordinates and metadata")
    print("4. Use tools like 'mcp' CLI or integrate into your application")

    print("\n🔗 Example tools that can connect:")
    print("- Claude Desktop (with MCP configuration)")
    print("- Custom MCP clients")
    print("- Command-line MCP tools")
    print("- VS Code extensions with MCP support")


if __name__ == "__main__":
    main()
