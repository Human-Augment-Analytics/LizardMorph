#!/usr/bin/env python3
"""
Check if the current CPU supports VNNI (Vector Neural Network Instructions).

VNNI is required for OpenVINO INT8 inference to be significantly faster than float32.
Without VNNI, INT8 inference may even be slower due to memory overhead.

Usage:
    python check_cpu_vnni.py

Output:
    Shows CPU model, instruction set support, and expected INT8 speedup.
"""

import os
import sys
import platform
import subprocess

try:
    from openvino import Core
except ImportError:
    print("Warning: openvino not installed, will use fallback CPU detection")
    Core = None


def check_vnni_linux():
    """Check VNNI support on Linux via /proc/cpuinfo."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('flags'):
                    flags = line.split(':', 1)[1].strip().split()
                    return 'vnni' in flags or 'avx512vnni' in flags
    except:
        return None


def check_vnni_macos():
    """Check VNNI support on macOS (mostly Intel, no VNNI on ARM)."""
    try:
        result = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True)
        cpu_model = result.strip()

        # Apple Silicon doesn't have VNNI, but has different INT8 capability
        if 'Apple' in cpu_model or 'M1' in cpu_model or 'M2' in cpu_model:
            return False  # ARM64, no VNNI but has other optimizations

        # Intel on macOS: check via sysctlbyname for AVX-512
        try:
            subprocess.check_output(['sysctl', '-n', 'hw.optional.avx512f'], text=True)
            return True
        except:
            return None
    except:
        return None


def check_vnni_windows():
    """Check VNNI support on Windows via cpuid."""
    try:
        import ctypes

        # This is a simplified check; full CPUID would require more complex code
        # For now, rely on OpenVINO detection or fallback
        return None
    except:
        return None


def get_cpu_info():
    """Get CPU model and vendor info."""
    if sys.platform == 'linux':
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        return line.split(':', 1)[1].strip()
        except:
            pass
    elif sys.platform == 'darwin':
        try:
            result = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True)
            return result.strip()
        except:
            pass
    elif sys.platform == 'win32':
        try:
            result = subprocess.check_output(['wmic', 'cpu', 'get', 'name'], text=True)
            lines = result.strip().split('\n')
            if len(lines) > 1:
                return lines[1].strip()
        except:
            pass

    return "Unknown"


def get_openvino_cpu_info():
    """Get detailed CPU info from OpenVINO."""
    if Core is None:
        return {}

    try:
        core = Core()
        metrics = core.get_property("CPU", "SUPPORTED_PROPERTIES")

        info = {}

        # Try to get CPU model
        try:
            cpu_model = core.get_property("CPU", "FULL_DEVICE_NAME")
            info["device_name"] = cpu_model
        except:
            pass

        # Get OPTIMIZATION_CAPABILITIES if available
        try:
            caps = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
            info["optimization_capabilities"] = caps
        except:
            pass

        return info
    except Exception as e:
        print(f"Warning: Could not query OpenVINO CPU info: {e}")
        return {}


def main():
    print("=" * 70)
    print("CPU VNNI (Vector Neural Network Instructions) Check")
    print("=" * 70)

    print(f"\nPlatform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")

    # Get CPU model
    cpu_model = get_cpu_info()
    print(f"CPU Model: {cpu_model}")

    # Check VNNI support
    print(f"\nChecking VNNI support...")

    vnni_supported = None

    if sys.platform == 'linux':
        vnni_supported = check_vnni_linux()
    elif sys.platform == 'darwin':
        vnni_supported = check_vnni_macos()
    elif sys.platform == 'win32':
        vnni_supported = check_vnni_windows()

    # Try OpenVINO detection
    ov_info = get_openvino_cpu_info()

    if "optimization_capabilities" in ov_info:
        caps = ov_info["optimization_capabilities"]
        print(f"OpenVINO capabilities: {caps}")

        # Check for VNNI-like optimizations
        caps_lower = str(caps).lower()
        if 'vnni' in caps_lower or 'avx512' in caps_lower:
            vnni_supported = True
        elif 'sse' in caps_lower or 'avx' in caps_lower:
            vnni_supported = False

    print(f"\nVNNI Support: ", end="")

    if vnni_supported is True:
        print("✓ YES")
    elif vnni_supported is False:
        print("✗ NO")
    else:
        print("? UNKNOWN (could not detect reliably)")

    # Estimate speedup
    print(f"\nEstimated INT8 Speedup:")

    if vnni_supported is True:
        print(f"  Expected: 2.0-3.5x faster than float32")
        print(f"  Status: OpenVINO INT8 RECOMMENDED ✓")
    elif vnni_supported is False:
        print(f"  Expected: 0.9-1.1x (no speedup, possible slowdown)")
        print(f"  Status: OpenVINO INT8 NOT RECOMMENDED ✗")
    else:
        print(f"  Expected: 2.0-3.5x (if VNNI) or 0.9-1.1x (if no VNNI)")
        print(f"  Status: UNCERTAIN - run actual benchmark to confirm")

    # CPU generation recommendations
    print(f"\nCPU Generation Support:")
    print(f"  VNNI support added in:")
    print(f"    - Intel Xeon: 2nd Gen (Cascade Lake, 2019) and newer ✓")
    print(f"    - Intel Xeon: 1st Gen (Skylake, 2017) - NO ✗")
    print(f"    - Intel Core: 10th Gen (Comet Lake, 2020) and newer ✓")
    print(f"    - AMD EPYC: 3rd Gen (Milan, 2021) and newer ✓")

    print(f"\nCurrent CPU generation (guessing from model name):")

    if "Xeon" in cpu_model:
        if "Cascade Lake" in cpu_model or "Ice Lake" in cpu_model or "Sapphire" in cpu_model:
            print(f"  → Likely supports VNNI (2019+) ✓")
        elif "Skylake" in cpu_model or "Platinum 8180" in cpu_model:
            print(f"  → May have VNNI (check /proc/cpuinfo)")
        else:
            print(f"  → Unknown generation, check /proc/cpuinfo")
    elif "Core i" in cpu_model:
        if any(x in cpu_model for x in ["10th Gen", "11th Gen", "12th Gen", "13th Gen"]):
            print(f"  → Likely supports VNNI ✓")
        else:
            print(f"  → May not have VNNI, check /proc/cpuinfo")
    elif "EPYC" in cpu_model:
        if "7002" in cpu_model:  # Milan
            print(f"  → Likely supports VNNI ✓")
        else:
            print(f"  → Check AMD specs for VNNI support")
    elif "Apple" in cpu_model:
        print(f"  → ARM64 (no VNNI, but has other INT8 optimizations)")

    # Next steps
    print(f"\nNext steps:")
    if vnni_supported is True:
        print(f"  1. Prepare calibration data (300+ toepad images)")
        print(f"  2. Run: python scripts/quantize_model_nncf.py \\")
        print(f"            --input models/yolo_obb_6class_h7.onnx \\")
        print(f"            --calibration-dir ./calibration_data \\")
        print(f"            --output models/yolo_obb_6class_h7_int8_nncf.xml")
        print(f"  3. Validate: python scripts/validate_int8_accuracy.py \\")
        print(f"               --fp32-model models/yolo_obb_6class_h7.onnx \\")
        print(f"               --int8-model models/yolo_obb_6class_h7_int8_nncf.xml \\")
        print(f"               --test-images ./test_images")
    elif vnni_supported is False:
        print(f"  1. Skip INT8 quantization (no performance benefit)")
        print(f"  2. Consider upgrading CPU to newer generation with VNNI")
        print(f"  3. Use float32 or other optimizations (pruning, distillation)")
    else:
        print(f"  1. Run actual benchmark to determine if INT8 helps:")
        print(f"     python scripts/benchmark_inference.py ./test_images.jpg")
        print(f"  2. Check /proc/cpuinfo manually: cat /proc/cpuinfo | grep vnni")
        print(f"  3. Based on results, proceed with INT8 or skip")

    print("\n" + "=" * 70)

    return 0 if vnni_supported is not False else 1


if __name__ == "__main__":
    sys.exit(main())
