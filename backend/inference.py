import os
import argparse
import utils
import visual_individual_performance

def run_hybrid_inference(img_path, body_model, scale_model, output_xml="inference_output.xml"):
    """Run the Hybrid Safe inference logic (Best Performance)."""
    print(f"Running Hybrid Inference on {img_path}...")
    print(f"Body Model: {body_model}")
    print(f"Scale Model: {scale_model}")
    
    utils.predictions_to_xml_dorsal_hybrid(
        img_path, 
        output_xml, 
        body_model, 
        scale_model
    )
    
    # Parse and print results
    data = visual_individual_performance.parse_xml_for_frontend(output_xml)
    coords = data.get("coords", [])
    print(f"Inference complete. Generated {len(coords)} landmarks.")
    print(f"Output saved to: {output_xml}")
    return coords

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Best-Performance Hybrid Inference for Lizard Landmarks")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--body", default="../models/lizard-x-ray/dorsal_predictor_clahe_best.dat", help="Path to body predictor (.dat)")
    parser.add_argument("--scale", default="../models/lizard-x-ray/scale_predictor_clahe.dat", help="Path to scale predictor (.dat)")
    parser.add_argument("--output", default="inference_results.xml", help="Path for output XML")
    args = parser.parse_args()
    
    # Check if models exist
    if not os.path.exists(args.body):
        print(f"Error: Body model not found at {args.body}")
    elif not os.path.exists(args.scale):
        print(f"Error: Scale model not found at {args.scale}")
    elif not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
    else:
        run_hybrid_inference(args.image, args.body, args.scale, args.output)
