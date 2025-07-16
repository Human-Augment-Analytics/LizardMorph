#!/usr/bin/env python3
"""
Startup script for LizardMorph MCP Server
"""

import os
import sys
from pathlib import Path


def setup_environment():
    """Set up the environment for the MCP server."""

    # Get paths
    mcp_dir = Path(__file__).parent
    backend_dir = mcp_dir.parent / "backend"
    predictor_file = backend_dir / "better_predictor_auto.dat"

    # Set environment variables
    os.environ.setdefault("PREDICTOR_FILE", str(predictor_file))

    # Check if predictor file exists
    if not predictor_file.exists():
        print(f"⚠️  Warning: Predictor file not found at: {predictor_file}")
        print("   The server will start but image processing will fail.")
        print("   Make sure the predictor file is available.")
    else:
        print(f"✅ Predictor file found: {predictor_file}")

    # Check if utils module is accessible
    if not (backend_dir / "utils.py").exists():
        print(f"❌ Error: utils.py not found at: {backend_dir}")
        print("   The server cannot function without the utils module.")
        sys.exit(1)
    else:
        print(f"✅ Utils module found: {backend_dir / 'utils.py'}")


def main():
    """Main startup function."""
    print("🚀 Starting LizardMorph MCP Server...")
    print("=" * 50)

    setup_environment()

    print("=" * 50)
    print("🔧 Environment configured successfully!")
    print("📡 Starting MCP server...")
    print("   (Use Ctrl+C to stop)")
    print("=" * 50)

    # Import and run the main server
    try:
        from main import main as server_main
        import asyncio

        asyncio.run(server_main())
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
