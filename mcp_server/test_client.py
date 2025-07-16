#!/usr/bin/env python3
"""
Test client for the LizardMorph MCP Server.
This script demonstrates how to interact with the MCP server.
"""

import asyncio
import base64
import json
from pathlib import Path


async def test_mcp_server():
    """Test the MCP server functionality."""
    print("🧪 Testing LizardMorph MCP Server...")

    # Test health check
    print("\n1. Testing health check...")
    health_request = {"tool": "health_check", "arguments": {}}
    print(f"Request: {json.dumps(health_request, indent=2)}")

    # Test predictor info
    print("\n2. Testing predictor info...")
    predictor_request = {"tool": "get_predictor_info", "arguments": {}}
    print(f"Request: {json.dumps(predictor_request, indent=2)}")

    # Test image processing (if sample image exists)
    sample_image_path = Path("../sample_image/0003_dorsal.jpg")
    if sample_image_path.exists():
        print("\n3. Testing image processing...")

        # Read and encode image
        with open(sample_image_path, "rb") as f:
            image_data = f.read()

        image_b64 = base64.b64encode(image_data).decode("utf-8")

        process_request = {
            "tool": "process_lizard_image",
            "arguments": {"image": image_b64, "image_name": "test_lizard.jpg"},
        }

        print(f"Request: process_lizard_image with image size: {len(image_data)} bytes")
        print(
            "📸 Image processing request prepared (base64 data truncated for display)"
        )

    else:
        print(f"\n3. ⚠️  Sample image not found at: {sample_image_path}")
        print("   To test image processing, place a lizard X-ray image there.")

    print("\n" + "=" * 50)
    print("📝 To actually run these tests:")
    print("1. Start the MCP server: python main.py")
    print("2. Connect via MCP client protocol")
    print("3. Send the JSON requests shown above")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
