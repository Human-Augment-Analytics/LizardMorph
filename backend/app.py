import utils
import os
import visual_individual_performance
import xray_preprocessing
from export_handler import ExportHandler
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS, cross_origin
import flask
from base64 import b64encode
import time
import random
import logging
import shutil
import json
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    static_folder="static",  # Specify the static folder explicitly
    static_url_path="",  # This makes static files available at root URL
)


CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",  # Allow all origins during development
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
        }
    },
)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "upload")
COLOR_CONTRAST_FOLDER = os.path.join(os.getcwd(), "color_constrasted")
TPS_DOWNLOAD_FOLDER = os.path.join(os.getcwd(), "tps_download")
IMAGE_DOWNLOAD_FOLDER = os.path.join(os.getcwd(), "image_download")
INVERT_IMAGE_FOLDER = os.path.join(os.getcwd(), "invert_image")
OUTPUTS_FOLDER = os.path.join(os.getcwd(), "outputs")

app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    COLOR_CONTRAST_FOLDER=COLOR_CONTRAST_FOLDER,
    TPS_DOWNLOAD_FOLDER=TPS_DOWNLOAD_FOLDER,
    IMAGE_DOWNLOAD_FOLDER=IMAGE_DOWNLOAD_FOLDER,
    INVERT_IMAGE_FOLDER=INVERT_IMAGE_FOLDER,
    OUTPUTS_FOLDER=OUTPUTS_FOLDER,
)

# Create folders if they don't exist
for folder in [
    UPLOAD_FOLDER,
    COLOR_CONTRAST_FOLDER,
    TPS_DOWNLOAD_FOLDER,
    IMAGE_DOWNLOAD_FOLDER,
    INVERT_IMAGE_FOLDER,
    OUTPUTS_FOLDER,
]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Initialize the export handler
export_handler = ExportHandler(OUTPUTS_FOLDER)


@app.route("/data", methods=["POST", "OPTIONS"])
def upload():
    if request.method == "OPTIONS":
        return "", 204

    try:
        images = request.files.getlist("image")
        all_data = []

        for image in images:
            if image:
                unique_name = f"{os.path.splitext(image.filename)[0]}_{int(time.time())}_{random.randint(1000,9999)}.jpg"
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
                image.save(image_path)

                processed_path = os.path.join(
                    COLOR_CONTRAST_FOLDER, f"processed_{unique_name}"
                )
                inverted_path = os.path.join(
                    INVERT_IMAGE_FOLDER, f"inverted_{unique_name}"
                )

                xray_preprocessing.process_single_image(image_path, processed_path)
                visual_individual_performance.invert_single_image(
                    image_path, inverted_path
                )

                # Generate the prediction XML
                xml_output_path = f"output_{unique_name}.xml"
                utils.predictions_to_xml_single(
                    "better_predictor_auto.dat", image_path, xml_output_path
                )

                # Generate CSV and TPS output files
                csv_output_path = f"output_{unique_name}.csv"
                tps_output_path = f"output_{unique_name}.tps"
                utils.dlib_xml_to_pandas(xml_output_path)
                utils.dlib_xml_to_tps(xml_output_path)

                # Create export directory and store files there
                export_dir = export_handler.create_export_directory(unique_name)

                # Copy original files to export directory
                export_handler.copy_file_to_export(image_path, export_dir)
                export_handler.copy_file_to_export(
                    processed_path, export_dir, f"processed_{unique_name}"
                )
                export_handler.copy_file_to_export(
                    inverted_path, export_dir, f"inverted_{unique_name}"
                )
                export_handler.copy_file_to_export(
                    xml_output_path, export_dir, f"{os.path.basename(xml_output_path)}"
                )
                export_handler.copy_file_to_export(
                    csv_output_path, export_dir, f"{os.path.basename(csv_output_path)}"
                )
                export_handler.copy_file_to_export(
                    tps_output_path, export_dir, f"{os.path.basename(tps_output_path)}"
                )

                # Parse XML for frontend
                data = visual_individual_performance.parse_xml_for_frontend(
                    xml_output_path
                )
                data["name"] = unique_name
                all_data.append(data)

                logger.info(f"Processed image: {unique_name}")

        return jsonify(all_data)

    except Exception as e:
        logger.error(f"Error in upload: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/image", methods=["POST"])
@cross_origin()
def get_input_image():
    image_filename = request.args.get("image_filename")
    if not image_filename:
        return jsonify({"error": "image_filename query parameter is required"}), 400

    image_data = {}

    # Define exact paths for each image version
    paths = {
        "image1": os.path.join(COLOR_CONTRAST_FOLDER, f"processed_{image_filename}"),
        "image2": os.path.join(INVERT_IMAGE_FOLDER, f"inverted_{image_filename}"),
        "image3": os.path.join(UPLOAD_FOLDER, image_filename),
    }

    for key, path in paths.items():
        if not os.path.exists(path):
            logger.warning(f"Image not found at {path}")
            # Don't fail - continue with other images
            image_data[key] = ""
            continue

        with open(path, "rb") as f:
            image_data[key] = b64encode(f.read()).decode("utf-8")

    # Check if we have at least the original image
    if not image_data["image3"]:
        return jsonify({"error": "Original image not found"}), 404

    return jsonify(image_data)


@app.route("/endpoint", methods=["POST"])
@cross_origin()
def process_scatter_data():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    coords = data.get("coords")
    name = data.get("name")

    if not coords or not name:
        return jsonify({"error": "Missing required data: coords or name"}), 400

    # Remove file extension if present
    base_name = name.split(".")[0] if "." in name else name

    # Create export directory for this output
    export_dir = export_handler.create_export_directory(name)

    # Create TPS file in both the download folder (legacy) and export directory
    tps_file_path = os.path.join(TPS_DOWNLOAD_FOLDER, f"{base_name}.tps")
    export_tps_path = export_handler.export_tps_file(coords, name, export_dir)

    logger.info(f"Creating TPS file at: {tps_file_path}")
    logger.info(f"Creating TPS file in export directory: {export_tps_path}")
    logger.info(f"Number of coordinates: {len(coords)}")
    logger.info(f"First coordinate: {coords[0] if coords else 'No coordinates'}")

    with open(tps_file_path, "w", encoding="utf-8", newline="\n") as tps_file:
        tps_file.write(f"LM={len(coords)}\n")
        # Get image height for y-flip
        from PIL import Image

        image_path = os.path.join(UPLOAD_FOLDER, name)
        with Image.open(image_path) as img:
            height = img.height

        # Write coordinates
        for point in coords:
            # Ensure points have x and y values
            if "x" in point and "y" in point:
                x = float(point["x"])
                y = float(point["y"])
                # Flip y to make origin bottom-left
                y_flipped = height - y
                tps_file.write(f"{x} {y_flipped}\n")
            else:
                logger.warning(f"Invalid point data: {point}")

        # Write image name without extension
        tps_file.write(f"IMAGE={base_name}")

    # Create annotated image
    try:
        logger.info(f"Creating annotated image for: {tps_file_path}")
        output_paths = visual_individual_performance.create_image(
            tps_file_path, IMAGE_DOWNLOAD_FOLDER
        )

        # Copy annotated images to export directory
        for path in output_paths:
            export_handler.copy_file_to_export(path, export_dir)

        # Construct URLs for the generated images
        image_urls = []
        if output_paths:
            for path in output_paths:
                image_urls.append(f"/api/images/{os.path.basename(path)}")

        logger.info(f"Annotated images created: {output_paths}")

        return jsonify(
            {
                "message": "TPS file and annotated image generated successfully",
                "tps_file": tps_file_path,
                "export_dir": export_dir,
                "image_paths": output_paths,
                "image_urls": image_urls,
            }
        )
    except Exception as e:
        logger.error(f"Error creating annotated image: {str(e)}", exc_info=True)
        return jsonify(
            {
                "message": "TPS file generated but image creation failed",
                "tps_file": tps_file_path,
                "export_dir": export_dir,
                "error": str(e),
            }
        )


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_DOWNLOAD_FOLDER, filename)


# New endpoint to list all files in the upload folder
@app.route("/list_uploads", methods=["GET"])
@cross_origin()
def list_uploads():
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            return jsonify([]), 200

        # Get all files from the upload folder with common image extensions
        valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}
        files = []

        for filename in os.listdir(UPLOAD_FOLDER):
            if os.path.isfile(os.path.join(UPLOAD_FOLDER, filename)):
                # Check if the file has a valid image extension
                _, ext = os.path.splitext(filename)
                if ext.lower() in valid_extensions:
                    files.append(filename)

        logger.info(f"Found {len(files)} files in upload folder")
        return jsonify(files), 200
    except Exception as e:
        logger.error(f"Error listing upload directory: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# Endpoint to process an existing image from the uploads folder
@app.route("/process_existing", methods=["POST"])
@cross_origin()
def process_existing():
    try:
        filename = request.args.get("filename")
        if not filename:
            return jsonify({"error": "filename parameter is required"}), 400

        image_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(image_path):
            return (
                jsonify({"error": f"File {filename} not found in uploads folder"}),
                404,
            )

        # Create export directory for this processing
        export_dir = export_handler.create_export_directory(filename)

        # Check if the processed versions already exist
        processed_path = os.path.join(COLOR_CONTRAST_FOLDER, f"processed_{filename}")
        inverted_path = os.path.join(INVERT_IMAGE_FOLDER, f"inverted_{filename}")
        xml_path = f"output_{filename}.xml"

        # Process the image if needed
        if not os.path.exists(processed_path):
            xray_preprocessing.process_single_image(image_path, processed_path)

        if not os.path.exists(inverted_path):
            visual_individual_performance.invert_single_image(image_path, inverted_path)

        # Generate XML if it doesn't exist
        if not os.path.exists(xml_path):
            utils.predictions_to_xml_single(
                "better_predictor_auto.dat", image_path, xml_path
            )
            utils.dlib_xml_to_pandas(xml_path)
            utils.dlib_xml_to_tps(xml_path)

        # Copy all files to export directory
        export_handler.copy_file_to_export(image_path, export_dir)
        export_handler.copy_file_to_export(
            processed_path, export_dir, f"processed_{filename}"
        )
        export_handler.copy_file_to_export(
            inverted_path, export_dir, f"inverted_{filename}"
        )
        export_handler.copy_file_to_export(
            xml_path, export_dir, os.path.basename(xml_path)
        )

        # Copy CSV and TPS files
        csv_path = f"output_{filename}.csv"
        tps_path = f"output_{filename}.tps"
        if os.path.exists(csv_path):
            export_handler.copy_file_to_export(
                csv_path, export_dir, os.path.basename(csv_path)
            )
        if os.path.exists(tps_path):
            export_handler.copy_file_to_export(
                tps_path, export_dir, os.path.basename(tps_path)
            )

        # Parse XML for frontend
        data = visual_individual_performance.parse_xml_for_frontend(xml_path)
        data["name"] = filename
        data["export_dir"] = export_dir

        logger.info(f"Processed existing image: {filename}")
        return jsonify(data)

    except Exception as e:
        logger.error(f"Error processing existing image: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# New endpoint to save annotations (updated landmark points)
@app.route("/save_annotations", methods=["POST"])
@cross_origin()
def save_annotations():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        coords = data.get("coords")
        name = data.get("name")

        if not coords or not name:
            return jsonify({"error": "Missing required data: coords or name"}), 400

        # Remove file extension if present for base name
        base_name = name.split(".")[0] if "." in name else name

        # Create timestamp for version control
        timestamp = int(time.time())

        # Define filenames for the output files
        xml_path = f"output_{name}.xml"
        csv_path = f"output_{name}.csv"
        tps_path = f"output_{name}.tps"

        # Create backup of existing files if they exist
        if os.path.exists(xml_path):
            shutil.copy2(xml_path, f"{xml_path}.{timestamp}.bak")
        if os.path.exists(csv_path):
            shutil.copy2(csv_path, f"{csv_path}.{timestamp}.bak")
        if os.path.exists(tps_path):
            shutil.copy2(tps_path, f"{tps_path}.{timestamp}.bak")

        # Create export directory for this save operation
        export_dir = export_handler.create_export_directory(
            f"{name}_updated_{timestamp}"
        )

        # Update XML file with new coordinates
        # Since we need direct XML manipulation, we'll read the current XML
        if os.path.exists(xml_path):
            try:
                import xml.etree.ElementTree as ET

                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Find the image element for our file
                for image in root.findall(".//image"):
                    if name in image.get("file", ""):
                        # Found the right image, now update the coordinates
                        for box in image.findall(".//box"):
                            # Remove existing parts
                            for part in box.findall("part"):
                                box.remove(part)

                            # Add new parts based on the provided coordinates
                            for i, point in enumerate(coords):
                                if "x" in point and "y" in point:
                                    part = ET.SubElement(box, "part")
                                    part.set("name", str(i))
                                    part.set("x", str(int(float(point["x"]))))
                                    part.set("y", str(int(float(point["y"]))))

                # Save the updated XML
                utils.pretty_xml(root, xml_path)

                # Now regenerate CSV and TPS files
                utils.dlib_xml_to_pandas(xml_path)
                utils.dlib_xml_to_tps(xml_path)

                logger.info(f"Updated XML, CSV, and TPS files for {name}")
            except Exception as xml_e:
                logger.error(f"Error updating XML file: {str(xml_e)}", exc_info=True)
                # If XML update fails, create TPS file directly
                logger.info(f"Falling back to direct TPS file creation")

        # Create TPS file directly from coordinates
        tps_file_path = os.path.join(TPS_DOWNLOAD_FOLDER, f"{base_name}.tps")
        export_tps_path = export_handler.export_tps_file(coords, name, export_dir)

        logger.info(f"Created TPS file at: {tps_file_path}")
        logger.info(f"Created TPS file in export directory: {export_tps_path}")

        # Generate annotated image with updated points
        try:
            output_paths = visual_individual_performance.create_image(
                tps_file_path, IMAGE_DOWNLOAD_FOLDER
            )

            # Copy all generated files to the export directory
            if os.path.exists(xml_path):
                export_handler.copy_file_to_export(
                    xml_path, export_dir, os.path.basename(xml_path)
                )
            if os.path.exists(csv_path):
                export_handler.copy_file_to_export(
                    csv_path, export_dir, os.path.basename(csv_path)
                )

            # Copy the annotated images
            for path in output_paths:
                export_handler.copy_file_to_export(path, export_dir)

            logger.info(f"Annotations saved successfully for {name}")

            return jsonify(
                {
                    "message": "Annotations saved successfully",
                    "export_dir": export_dir,
                    "files": {
                        "xml": xml_path if os.path.exists(xml_path) else None,
                        "csv": csv_path if os.path.exists(csv_path) else None,
                        "tps": tps_path if os.path.exists(tps_path) else None,
                        "images": output_paths,
                    },
                }
            )

        except Exception as img_e:
            logger.error(f"Error creating annotated image: {str(img_e)}", exc_info=True)
            return jsonify(
                {
                    "message": "Annotations saved but image creation failed",
                    "export_dir": export_dir,
                    "error": str(img_e),
                }
            )

    except Exception as e:
        logger.error(f"Error saving annotations: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# New endpoint to create zip from all export directories
@app.route("/download_all", methods=["GET"])
@cross_origin()
def download_all_exports():
    try:
        # Get all export directories
        export_dirs = export_handler.collect_all_export_directories()

        if not export_dirs:
            return jsonify({"error": "No exports found to download"}), 404

        # Create a zip file containing all export directories
        zip_path = export_handler.create_export_zip(export_dirs)

        if not zip_path:
            return jsonify({"error": "Failed to create zip file"}), 500

        # Return the file for download
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=os.path.basename(zip_path),
            mimetype="application/zip",
        )

    except Exception as e:
        logger.error(f"Error creating zip file: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


# Add these routes to serve the React frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join("static", path)):
        return send_from_directory("static", path)
    else:
        return send_from_directory("static", "index.html")


# Make sure your app runs on the correct host and port if started directly
if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 5000))
    app.run(host="0.0.0.0", port=port)
