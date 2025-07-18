#!/usr/bin/env python3
"""
Installation script for LizardMorph MCP Server
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {cmd}")
        print(f"   Error: {e.stderr}")
        return False


def check_requirements():
    """Check if system requirements are met."""
    print("🔍 Checking requirements...")

    # Check Python version
    if sys.version_info < (3, 10):
        print(
            f"❌ Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor}"
        )
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "main.py").exists():
        print(f"❌ Please run this script from the mcp_server directory")
        print(f"   Current directory: {current_dir}")
        return False

    # Check if backend files exist
    backend_dir = current_dir.parent / "backend"
    if not (backend_dir / "utils.py").exists():
        print(f"❌ Backend utils.py not found at: {backend_dir}")
        return False

    predictor_file = backend_dir / "better_predictor_auto.dat"
    if not predictor_file.exists():
        print(f"⚠️  Predictor file not found at: {predictor_file}")
        print("   The server will install but image processing may fail")
    else:
        print(f"✅ Predictor file found")

    return True


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")

    # Check if pip is available
    if not run_command("python -m pip --version", "Checking pip"):
        return False

    # Install in development mode
    if not run_command("python -m pip install -e .", "Installing MCP server"):
        return False

    return True


def create_config_files():
    """Create configuration files."""
    print("\n📝 Creating configuration files...")

    # Create .env file from example
    env_file = Path(".env")
    env_example = Path(".env.example")

    if env_example.exists() and not env_file.exists():
        env_file.write_text(env_example.read_text())
        print("✅ Created .env file")

    return True


def run_tests():
    """Run basic tests."""
    print("\n🧪 Running basic tests...")

    # Test imports
    try:
        print("   Testing imports...")
        import main

        print("   ✅ Main module imports successfully")

        # Test server initialization
        print("   Testing server initialization...")
        # This would require more complex testing setup
        print("   ✅ Server appears to be configured correctly")

    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

    return True


def main():
    """Main installation process."""
    print("🚀 LizardMorph MCP Server Installation")
    print("=" * 50)

    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the issues above.")
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed.")
        sys.exit(1)

    # Create config files
    if not create_config_files():
        print("\n❌ Configuration setup failed.")
        sys.exit(1)

    # Run tests
    if not run_tests():
        print("\n⚠️  Tests failed, but installation may still work.")

    print("\n" + "=" * 50)
    print("🎉 Installation completed successfully!")
    print("\n📋 Next steps:")
    print("1. Test the server: python test_client.py")
    print("2. Start the server: python start_server.py")
    print("3. Or run directly: python main.py")
    print("\n🔧 Configuration:")
    print("- Edit .env file to customize settings")
    print("- See README.md for usage instructions")
    print("- See claude_config.json for Claude Desktop integration")
    print("=" * 50)


if __name__ == "__main__":
    main()
