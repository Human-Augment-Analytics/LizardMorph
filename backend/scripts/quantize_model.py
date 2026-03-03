#!/usr/bin/env python3
"""Quantize YOLO ONNX model from float32 to INT8 using dynamic quantization."""

import argparse
import os
import sys

from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    parser = argparse.ArgumentParser(description="Quantize YOLO ONNX model to INT8")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "../../models/lizard-toe-pad/yolo_obb_6class_h7.onnx"),
        help="Path to input float32 ONNX model",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output quantized model (default: <input>_int8.onnx)",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: Input model not found: {input_path}")
        sys.exit(1)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_int8{ext}"

    input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Input model: {input_path} ({input_size_mb:.1f} MB)")
    print(f"Output model: {output_path}")
    print("Quantizing with dynamic INT8...")

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
    )

    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! Output: {output_path} ({output_size_mb:.1f} MB)")
    print(f"Size reduction: {input_size_mb:.1f} MB -> {output_size_mb:.1f} MB ({(1 - output_size_mb / input_size_mb) * 100:.1f}% smaller)")


if __name__ == "__main__":
    main()
