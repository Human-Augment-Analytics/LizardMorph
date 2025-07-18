#!/usr/bin/env python3
"""Enhanced test script with better error handling and debugging"""

import json
import subprocess
import sys
import base64
import os
import time
import threading
import queue
from pathlib import Path


def read_with_timeout(proc, timeout=5):
    """Read from subprocess with timeout - Windows compatible"""

    def read_line(proc, q):
        try:
            line = proc.stdout.readline()
            q.put(line)
        except Exception as e:
            q.put(f"ERROR: {e}")

    q = queue.Queue()
    thread = threading.Thread(target=read_line, args=(proc, q))
    thread.daemon = True
    thread.start()

    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


def test_server_startup():
    """Test just the server startup without trying to communicate"""
    print("\n🔍 Testing server startup behavior...")

    main_file = Path(__file__).parent / "main.py"

    try:
        # Start the server and capture initial output
        print("🚀 Starting server process...")
        proc = subprocess.Popen(
            [sys.executable, str(main_file)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent,
        )

        print(f"✅ Process started with PID: {proc.pid}")

        # Wait a bit and capture any initial output
        print("⏰ Waiting 5 seconds for startup output...")
        time.sleep(5)

        # Check if process is still running
        if proc.poll() is not None:
            print(f"❌ Process exited with code: {proc.poll()}")
            stdout, stderr = proc.communicate()
            print(f"📤 STDOUT:\n{stdout}")
            print(f"📥 STDERR:\n{stderr}")
            return False

        print("✅ Process is still running after 5 seconds")

        # Try to read any stderr output (where debug messages should be)
        print("📖 Checking for startup debug messages...")
        startup_messages = []
        for i in range(10):  # Try to read multiple lines from stderr
            try:
                # Read from stderr instead of stdout
                def read_stderr():
                    try:
                        return proc.stderr.readline()
                    except:
                        return ""

                import threading

                stderr_queue = queue.Queue()
                stderr_thread = threading.Thread(
                    target=lambda: stderr_queue.put(read_stderr())
                )
                stderr_thread.daemon = True
                stderr_thread.start()
                stderr_thread.join(timeout=0.5)

                if not stderr_queue.empty():
                    line = stderr_queue.get()
                    if line and line.strip():
                        startup_messages.append(line.strip())
                        print(f"📄 Debug line {i+1}: {repr(line.strip())}")
                    else:
                        break
                else:
                    break
            except Exception as e:
                print(f"⚠️ Error reading debug line {i+1}: {e}")
                break

        if not startup_messages:
            print("📄 No startup debug messages received")
        else:
            print(f"✅ Received {len(startup_messages)} startup debug messages")

        # Terminate the process
        print("🛑 Terminating process...")
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=5)
            if stdout.strip():
                print(f"📤 Final STDOUT:\n{stdout}")
            if stderr.strip():
                print(f"📥 Final STDERR:\n{stderr}")
        except subprocess.TimeoutExpired:
            print("⏰ Timeout getting final output, killing process...")
            proc.kill()
            proc.wait()

        return True

    except Exception as e:
        print(f"❌ Server startup test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_basic_communication():
    """Test basic stdin/stdout communication with the server"""
    print("\n💬 Testing basic communication...")

    main_file = Path(__file__).parent / "main.py"

    try:
        proc = subprocess.Popen(
            [sys.executable, str(main_file)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent,
        )

        print(f"✅ Process started with PID: {proc.pid}")

        # Wait for startup
        time.sleep(2)

        if proc.poll() is not None:
            print(f"❌ Process died during startup: {proc.poll()}")
            return False

        # Send a simple test message (not valid JSON-RPC, just to test communication)
        print("📤 Sending test message...")
        try:
            proc.stdin.write("test message\n")
            proc.stdin.flush()
            print("✅ Test message sent")
        except Exception as e:
            print(f"❌ Failed to send test message: {e}")
            return False

        # Try to read any response
        print("📖 Reading response...")
        response = read_with_timeout(proc, timeout=3)
        if response:
            print(f"📄 Response: {repr(response)}")
        else:
            print("📄 No response received")

        # Cleanup
        proc.terminate()
        proc.wait(timeout=5)

        return True

    except Exception as e:
        print(f"❌ Basic communication test failed: {e}")
        return False


def test_direct_import():
    """Test if we can import and run the server code directly"""
    print("\n🔬 Testing direct import...")
    try:
        # Try to import the main module
        import sys

        sys.path.insert(0, str(Path(__file__).parent))

        print("📦 Attempting to import main module...")
        import main

        print("✅ Main module imported successfully")

        # Check if key components are available
        print("🔍 Checking key components...")
        if hasattr(main, "processor"):
            print("✅ Processor available")
        else:
            print("❌ Processor not available")

        if hasattr(main, "server"):
            print("✅ Server available")
        else:
            print("❌ Server not available")

        # Try to create test objects
        print("🧪 Testing MCP object creation...")
        from mcp.types import TextContent, CallToolResult

        test_text = TextContent(type="text", text="test")
        print(f"✅ TextContent created: {test_text}")

        test_result = CallToolResult(content=[test_text], isError=False)
        print(f"✅ CallToolResult created: {test_result}")

        return True

    except Exception as e:
        print(f"❌ Direct import test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_mcp_server():
    """Test the MCP server with enhanced error handling"""
    print("\n🧪 Enhanced Testing of MCP Server...")

    # First, check if the main.py file exists
    main_file = Path(__file__).parent / "main.py"
    if not main_file.exists():
        print(f"❌ main.py not found at {main_file}")
        return False

    print(f"✅ Found main.py at {main_file}")

    proc = None
    try:
        # Start the MCP server with enhanced error capture
        print("🚀 Starting MCP server process...")
        proc = subprocess.Popen(
            [sys.executable, str(main_file)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent,
            bufsize=0,  # Unbuffered
        )

        print(f"✅ Process started with PID: {proc.pid}")

        # Give the server a moment to start up
        time.sleep(2)

        # Check if process is still running
        if proc.poll() is not None:
            print(f"❌ Process exited early with code: {proc.poll()}")
            stdout, stderr = proc.communicate()
            print(f"📤 STDOUT: {stdout}")
            print(f"📥 STDERR: {stderr}")
            return False

        print("✅ Process is running, attempting communication...")

        # Step 1: Initialize the MCP server
        init_request = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        print("📤 Sending initialization request...")
        try:
            request_line = json.dumps(init_request) + "\n"
            print(f"📝 Request: {request_line.strip()}")
            proc.stdin.write(request_line)
            proc.stdin.flush()
            print("✅ Request sent successfully")
        except Exception as e:
            print(f"❌ Failed to send initialization request: {e}")
            return False

        # Read initialization response with timeout
        print("⏰ Waiting for initialization response...")
        try:
            init_response = read_with_timeout(proc, timeout=10)
            if init_response is None:
                print("❌ Timeout waiting for initialization response")
                return False

            print(f"📥 Init response: {init_response.strip()}")

            # Try to parse the response
            try:
                response_data = json.loads(init_response)
                if "result" in response_data:
                    print("✅ Initialization successful")
                else:
                    print(f"⚠️ Unexpected initialization response: {response_data}")
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse initialization response: {e}")
                print(f"📄 Raw response: {repr(init_response)}")
                return False

        except Exception as e:
            print(f"❌ Error reading initialization response: {e}")
            return False

        # Step 2: Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }

        print("📤 Sending initialized notification...")
        try:
            notification_line = json.dumps(initialized_notification) + "\n"
            proc.stdin.write(notification_line)
            proc.stdin.flush()
            print("✅ Notification sent successfully")
        except Exception as e:
            print(f"❌ Failed to send initialized notification: {e}")
            return False

        # Step 3: Test health_check tool
        health_check_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "health_check", "arguments": {}},
        }

        print("📤 Sending health check request...")
        try:
            health_line = json.dumps(health_check_request) + "\n"
            print(f"📝 Health check request: {health_line.strip()}")
            proc.stdin.write(health_line)
            proc.stdin.flush()
            print("✅ Health check request sent successfully")
        except Exception as e:
            print(f"❌ Failed to send health check request: {e}")
            return False

        # Read health check response with timeout
        print("⏰ Waiting for health check response...")
        try:
            health_response = read_with_timeout(proc, timeout=15)
            if health_response is None:
                print("❌ Timeout waiting for health check response")
                return False

            print(f"📥 Health check response: {health_response.strip()}")

            # Parse and validate response
            try:
                response_data = json.loads(health_response)
                if "result" in response_data and "content" in response_data["result"]:
                    print("✅ MCP server health check passed!")
                    print(
                        f"📊 Response content: {response_data['result']['content'][0]['text']}"
                    )
                    return True
                else:
                    print("❌ Invalid health check response format")
                    print(f"📄 Full response: {response_data}")
                    return False

            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse health check response: {e}")
                print(f"📄 Raw response: {repr(health_response)}")
                return False

        except Exception as e:
            print(f"❌ Error reading health check response: {e}")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ MCP server test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if proc:
            try:
                print("🧹 Cleaning up process...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    print("✅ Process terminated cleanly")
                except subprocess.TimeoutExpired:
                    print("⚠️ Process didn't terminate, killing...")
                    proc.kill()
                    proc.wait()
                    print("💀 Process killed")
            except Exception as cleanup_error:
                print(f"⚠️ Error during cleanup: {cleanup_error}")


if __name__ == "__main__":
    print("🚀 Running enhanced MCP server tests...")

    # Test 1: Server startup behavior
    startup_success = test_server_startup()

    # Test 2: Basic communication
    comm_success = test_basic_communication()

    # Test 3: Direct import
    import_success = test_direct_import()

    # Test 4: Full MCP communication (only if basic comm works)
    subprocess_success = False
    if comm_success:
        subprocess_success = test_mcp_server()
    else:
        print("⏭️ Skipping MCP test due to communication issues")

    print(f"\n📊 Test Results:")
    print(f"   Server startup: {'✅ PASS' if startup_success else '❌ FAIL'}")
    print(f"   Basic communication: {'✅ PASS' if comm_success else '❌ FAIL'}")
    print(f"   Direct import: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"   MCP protocol: {'✅ PASS' if subprocess_success else '❌ FAIL'}")

    overall_success = import_success and subprocess_success
    print(f"   Overall: {'✅ PASS' if overall_success else '❌ FAIL'}")

    sys.exit(0 if overall_success else 1)
