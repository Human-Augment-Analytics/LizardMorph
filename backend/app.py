import utils
import visual_individual_performance
import xray_preprocessing
from export_handler import ExportHandler
from session_manager import SessionManager

import os
import hmac
import hashlib
import subprocess
import json
from flask import Flask, jsonify, request, send_from_directory, send_file, session
from flask_cors import CORS, cross_origin
from base64 import b64encode
import time
import logging
import shutil
from dotenv import load_dotenv
import psutil
import threading
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Load ENV variables
load_dotenv()
frontend_dir = os.getenv("FRONTEND_DIR", "../frontend/dist")
session_dir = os.getenv("SESSION_DIR", "sessions")

# Model files for different view types
DORSAL_PREDICTOR_FILE = os.getenv("DORSAL_PREDICTOR_FILE", "../models/lizard-x-ray/better_predictor_auto.dat")
LATERAL_PREDICTOR_FILE = os.getenv("LATERAL_PREDICTOR_FILE", "../models/lizard-x-ray/lateral_predictor_auto.dat")
TOEPADS_PREDICTOR_FILE = os.getenv("TOEPADS_PREDICTOR_FILE", "./toepads_predictor_auto.dat")
CUSTOM_PREDICTOR_FILE = os.getenv("CUSTOM_PREDICTOR_FILE", "./custom_predictor_auto.dat")

# Detector files for different view types (dlib fhog object detectors)
DORSAL_DETECTOR_FILE = os.getenv("DORSAL_DETECTOR_FILE", None)
LATERAL_DETECTOR_FILE = os.getenv("LATERAL_DETECTOR_FILE", None)
TOEPADS_DETECTOR_FILE = os.getenv("TOEPADS_DETECTOR_FILE", None)
CUSTOM_DETECTOR_FILE = os.getenv("CUSTOM_DETECTOR_FILE", None)

# Lizard Toepad model files
# Note: To use cropped predictors (e.g., toe_predictor_cropped.dat), set the environment variable:
# TOEPAD_TOE_PREDICTOR="../models/lizard-toe-pad/toe_predictor_cropped.dat"
# The inference code will automatically detect cropped predictors by filename and apply appropriate preprocessing.
TOEPAD_YOLO_MODEL = os.getenv("TOEPAD_YOLO_MODEL", "../models/lizard-toe-pad/yolo_bounding_box.pt")
TOEPAD_TOE_PREDICTOR = os.getenv("TOEPAD_TOE_PREDICTOR", "../models/lizard-toe-pad/lizard_toe.dat")
TOEPAD_SCALE_PREDICTOR = os.getenv("TOEPAD_SCALE_PREDICTOR", "../models/lizard-toe-pad/lizard_scale.dat")
TOEPAD_FINGER_PREDICTOR = os.getenv("TOEPAD_FINGER_PREDICTOR", "../models/lizard-toe-pad/lizard_finger.dat")



# Default predictor file (fallback)
predictor_file = os.getenv("PREDICTOR_FILE", "../models/lizard-x-ray/better_predictor_auto.dat")

# Webhook configuration
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "your-webhook-secret-here")
REPO_NAME = os.getenv("REPO_NAME", "LizardMorph")
MAIN_BRANCH = os.getenv("MAIN_BRANCH", "main")
VERIFY_SIGNATURE = os.getenv("VERIFY_SIGNATURE", "true").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info(f"Default predictor file: {predictor_file}")
logger.info(f"Dorsal predictor file: {DORSAL_PREDICTOR_FILE}")
logger.info(f"Lateral predictor file: {LATERAL_PREDICTOR_FILE}")
logger.info(f"Toepads predictor file: {TOEPADS_PREDICTOR_FILE}")
logger.info(f"Custom predictor file: {CUSTOM_PREDICTOR_FILE}")
logger.info(f"Dorsal detector file: {DORSAL_DETECTOR_FILE}")
logger.info(f"Lateral detector file: {LATERAL_DETECTOR_FILE}")
logger.info(f"Toepads detector file: {TOEPADS_DETECTOR_FILE}")
logger.info(f"Custom detector file: {CUSTOM_DETECTOR_FILE}")
logger.info(f"Toepad YOLO model: {TOEPAD_YOLO_MODEL}")
logger.info(f"Toepad toe predictor: {TOEPAD_TOE_PREDICTOR}")
logger.info(f"Toepad scale predictor: {TOEPAD_SCALE_PREDICTOR}")
logger.info(f"Toepad finger predictor: {TOEPAD_FINGER_PREDICTOR}")


app = Flask(__name__)

# Configure Flask session
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
app.config["SESSION_TYPE"] = "filesystem"


CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",  # Allow all origins during development
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "X-Session-ID"],
        }
    },
)

SESSIONS_FOLDER = os.path.join(os.getcwd(), session_dir)
UPLOAD_FOLDER = os.path.join(os.getcwd(), "upload")
COLOR_CONTRAST_FOLDER = os.path.join(os.getcwd(), "color_constrasted")
TPS_DOWNLOAD_FOLDER = os.path.join(os.getcwd(), "tps_download")
IMAGE_DOWNLOAD_FOLDER = os.path.join(os.getcwd(), "image_download")
INVERT_IMAGE_FOLDER = os.path.join(os.getcwd(), "invert_image")
OUTPUTS_FOLDER = os.path.join(os.getcwd(), "outputs")

app.config.update(
    SESSIONS_FOLDER=SESSIONS_FOLDER,
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    COLOR_CONTRAST_FOLDER=COLOR_CONTRAST_FOLDER,
    TPS_DOWNLOAD_FOLDER=TPS_DOWNLOAD_FOLDER,
    IMAGE_DOWNLOAD_FOLDER=IMAGE_DOWNLOAD_FOLDER,
    INVERT_IMAGE_FOLDER=INVERT_IMAGE_FOLDER,
    OUTPUTS_FOLDER=OUTPUTS_FOLDER,
)

# Create folders if they don't exist
for folder in [
    SESSIONS_FOLDER,
    UPLOAD_FOLDER,
    COLOR_CONTRAST_FOLDER,
    TPS_DOWNLOAD_FOLDER,
    IMAGE_DOWNLOAD_FOLDER,
    INVERT_IMAGE_FOLDER,
    OUTPUTS_FOLDER,
]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Initialize the export handler and session manager
export_handler = ExportHandler(OUTPUTS_FOLDER)
session_manager = SessionManager(SESSIONS_FOLDER)

# Initialize Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('memory_usage_percent', 'Memory usage percentage')
DISK_USAGE = Gauge('disk_usage_percent', 'Disk usage percentage')

# Background thread to update system metrics with memory optimization
def update_system_metrics():
    while True:
        try:
            # Use psutil with memory optimization
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            CPU_USAGE.set(cpu_percent)
            MEMORY_USAGE.set(memory.percent)
            DISK_USAGE.set(disk.percent)
            
            # Force garbage collection after metrics update
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Error updating system metrics: {e}")
        time.sleep(30)  # Reduced frequency to save memory

# Start system metrics collection
metrics_thread = threading.Thread(target=update_system_metrics, daemon=True)
metrics_thread.start()

# Prometheus metrics endpoint
@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

# System metrics endpoint
@app.route('/system/metrics')
def system_metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

# Decorator to track metrics for all endpoints
def track_metrics(f):
    def wrapper(*args, **kwargs):
        from flask import request
        start_time = time.time()
        try:
            response = f(*args, **kwargs)
            status = response[1] if isinstance(response, tuple) else 200
            REQUEST_COUNT.labels(method=request.method, endpoint=request.endpoint, status=status).inc()
            REQUEST_LATENCY.labels(method=request.method, endpoint=request.endpoint).observe(time.time() - start_time)
            return response
        except Exception as e:
            REQUEST_COUNT.labels(method=request.method, endpoint=request.endpoint, status=500).inc()
            raise e
    wrapper.__name__ = f.__name__
    return wrapper

# Decorator to restrict system endpoints to localhost only
def localhost_only(f):
    def wrapper(*args, **kwargs):
        from flask import request, jsonify
        
        # Get the real client IP considering proxies
        client_ip = request.remote_addr
        
        # Check various headers for the real client IP
        x_forwarded_for = request.headers.get('X-Forwarded-For', '')
        x_real_ip = request.headers.get('X-Real-IP', '')
        
        # If we have forwarded headers, use the first IP
        if x_forwarded_for:
            client_ip = x_forwarded_for.split(',')[0].strip()
        elif x_real_ip:
            client_ip = x_real_ip
        
        # Only allow localhost IPs
        allowed_ips = ['127.0.0.1', 'localhost', '::1']
        
        if client_ip not in allowed_ips:
            return jsonify({"error": "Access denied - localhost only"}), 403
        
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

# System monitoring functions
def get_cpu_usage():
    """Get current CPU usage percentage"""
    try:
        return psutil.cpu_percent(interval=1)
    except Exception as e:
        logger.error(f"Error getting CPU usage: {e}")
        return 0

def get_memory_usage():
    """Get current memory usage percentage"""
    try:
        memory = psutil.virtual_memory()
        return memory.percent
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}")
        return 0

def get_disk_usage():
    """Get current disk usage percentage"""
    try:
        disk = psutil.disk_usage('/')
        return (disk.used / disk.total) * 100
    except Exception as e:
        logger.error(f"Error getting disk usage: {e}")
        return 0

def get_system_info():
    """Get comprehensive system information"""
    try:
        memory = psutil.virtual_memory()
        return {
            'cpu_percent': get_cpu_usage(),
            'memory_percent': get_memory_usage(),
            'disk_percent': get_disk_usage(),
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'memory_total_gb': round(memory.total / (1024**3), 2),
            'memory_used_gb': round(memory.used / (1024**3), 2),
            'cpu_count': psutil.cpu_count(),
            'load_average': psutil.getloadavg()
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {}

def cleanup_memory():
    """Force memory cleanup"""
    try:
        import gc
        gc.collect()
        
        # Get memory info before and after cleanup
        memory_before = psutil.virtual_memory()
        gc.collect()
        memory_after = psutil.virtual_memory()
        
        freed_mb = (memory_before.used - memory_after.used) / (1024 * 1024)
        logger.info(f"Memory cleanup freed {freed_mb:.2f} MB")
        
        return freed_mb
    except Exception as e:
        logger.error(f"Error during memory cleanup: {e}")
        return 0


def get_session_id():
    """Get or create session ID for the current request."""
    session_id = request.headers.get("X-Session-ID")
    if not session_id:
        # Try to get from session cookie as fallback
        session_id = session.get("session_id")

    if not session_id:
        # Create new session
        session_id = session_manager.create_session()
        session["session_id"] = session_id

    return session_id


def get_session_folders(session_id):
    """Get session-specific folder paths."""
    session_data = session_manager.get_session(session_id)
    if not session_data:
        # Create session if it doesn't exist
        session_id = session_manager.create_session(session_id)
        session_data = session_manager.get_session(session_id)

    return session_data


def get_view_type_config(view_type):
    """
    Get the appropriate predictor file and detector file based on view type.
    
    Args:
        view_type (str): The view type (dorsal, lateral, toepads, toepad, custom)
        
    Returns:
        tuple: (predictor_file_path, detector_file_path, yolo_model_path)
               For non-toepad types, yolo_model_path will be None
    """
    view_type = view_type.lower() if view_type else "dorsal"
    
    if view_type == "dorsal":
        predictor_file = DORSAL_PREDICTOR_FILE
        detector_file = DORSAL_DETECTOR_FILE
        yolo_model = None
    elif view_type == "lateral":
        predictor_file = LATERAL_PREDICTOR_FILE
        detector_file = LATERAL_DETECTOR_FILE
        yolo_model = None
    elif view_type == "toepads":
        predictor_file = TOEPADS_PREDICTOR_FILE
        detector_file = TOEPADS_DETECTOR_FILE
        yolo_model = None
    elif view_type == "toepad":
        # For toepad, we use YOLO + dlib predictors
        # Default to toe predictor, but can be specified per request
        predictor_file = TOEPAD_TOE_PREDICTOR
        detector_file = None
        yolo_model = TOEPAD_YOLO_MODEL
    elif view_type == "custom":
        predictor_file = CUSTOM_PREDICTOR_FILE
        detector_file = CUSTOM_DETECTOR_FILE
        yolo_model = None
    else:
        # Default to dorsal if unknown view type
        logger.warning(f"Unknown view type '{view_type}', using dorsal configuration")
        predictor_file = DORSAL_PREDICTOR_FILE
        detector_file = DORSAL_DETECTOR_FILE
        yolo_model = None
    
    logger.info(f"View type '{view_type}' configured with:")
    logger.info(f"  - Predictor file: {predictor_file}")
    logger.info(f"  - Detector file: {detector_file}")
    logger.info(f"  - YOLO model: {yolo_model}")
    
    return predictor_file, detector_file, yolo_model


def cleanup_on_startup():
    """
    Optional cleanup function that can be called on app startup
    to clear any leftover files from previous sessions.
    """
    try:
        logger.info("Performing startup cleanup...")
        directories_to_check = [
            UPLOAD_FOLDER,
            COLOR_CONTRAST_FOLDER,
            INVERT_IMAGE_FOLDER,
            TPS_DOWNLOAD_FOLDER,
            IMAGE_DOWNLOAD_FOLDER,
        ]

        for directory in directories_to_check:
            if os.path.exists(directory):
                file_count = len(
                    [
                        f
                        for f in os.listdir(directory)
                        if os.path.isfile(os.path.join(directory, f))
                    ]
                )
                if file_count > 0:
                    logger.info(f"Found {file_count} files in {directory}")

        logger.info("Startup cleanup completed")
    except Exception as e:
        logger.error(f"Error during startup cleanup: {str(e)}")


# Uncomment the next line if you want to clear files on every app startup
# cleanup_on_startup()


@app.route("/session/start", methods=["POST"])
@cross_origin()
@track_metrics
def start_session():
    """Start a new session and return the session ID."""
    try:
        session_id = get_session_id()
        session_id = session_manager.create_session(session_id)
        session["session_id"] = session_id

        logger.info(f"Started new session: {session_id}")

        return (
            jsonify(
                {
                    "success": True,
                    "session_id": session_id,
                    "message": "New session started successfully",
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error starting session: {str(e)}", exc_info=True)
        return (
            jsonify({"success": False, "error": f"Failed to start session: {str(e)}"}),
            500,
        )


@app.route("/session/info", methods=["GET"])
@cross_origin()
@track_metrics
def get_session_info():
    """Get information about the current session."""
    try:
        session_id = get_session_id()
        session_data = session_manager.get_session(session_id)

        if not session_data:
            return jsonify({"success": False, "error": "Session not found"}), 404

        # Count files in session
        file_count = 0
        session_folder = session_data["session_folder"]
        if os.path.exists(session_folder):
            for root, dirs, files in os.walk(session_folder):
                file_count += len(files)

        return (
            jsonify(
                {
                    "success": True,
                    "session_id": session_id,
                    "session_id_short": session_id[:8],
                    "created_at": session_data["created_at"],
                    "session_folder": session_data["session_folder"],
                    "file_count": file_count,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error getting session info: {str(e)}", exc_info=True)
        return (
            jsonify(
                {"success": False, "error": f"Failed to get session info: {str(e)}"}
            ),
            500,
        )


@app.route("/session/list", methods=["GET"])
@cross_origin()
@track_metrics
def list_sessions():
    """List all available sessions."""
    try:
        sessions = session_manager.list_sessions()

        return jsonify({"success": True, "sessions": sessions}), 200

    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}", exc_info=True)
        return (
            jsonify({"success": False, "error": f"Failed to list sessions: {str(e)}"}),
            500,
        )


@app.route("/data", methods=["POST", "OPTIONS"])
@track_metrics
def upload():
    if request.method == "OPTIONS":
        return "", 204

    try:
        # Get session-specific folders
        session_id = get_session_id()
        session_data = get_session_folders(session_id)

        # Get view type from form data
        view_type = request.form.get("view_type", "dorsal")
        # Get toepad predictor type if specified (toe, scale, finger)
        toepad_predictor_type = request.form.get("toepad_predictor_type", "toe")
        predictor_file_path, detector_file_path, yolo_model_path = get_view_type_config(view_type)
        
        # For toepad view type, select the appropriate predictor
        if view_type.lower() == "toepad":
            if toepad_predictor_type == "scale":
                predictor_file_path = TOEPAD_SCALE_PREDICTOR
            elif toepad_predictor_type == "finger":
                predictor_file_path = TOEPAD_FINGER_PREDICTOR
            else:  # default to toe
                predictor_file_path = TOEPAD_TOE_PREDICTOR
        
        logger.info(f"Processing images with view type: {view_type}, predictor: {predictor_file_path}, detector: {detector_file_path}, YOLO: {yolo_model_path}")

        images = request.files.getlist("image")
        all_data = []

        for image in images:
            if image:
                unique_name = f"{os.path.splitext(image.filename)[0]}.jpg"

                # Use session-specific folders
                image_path = os.path.join(session_data["upload_folder"], unique_name)
                processed_path = os.path.join(
                    session_data["processed_folder"], f"processed_{unique_name}"
                )
                inverted_path = os.path.join(
                    session_data["inverted_folder"], f"inverted_{unique_name}"
                )

                image.save(image_path)

                # Process images with memory cleanup
                try:
                    xray_preprocessing.process_single_image(image_path, processed_path)
                    visual_individual_performance.invert_single_image(
                        image_path, inverted_path
                    )

                    # Generate the prediction XML in the session outputs folder
                    xml_output_path = os.path.join(
                        session_data["outputs_folder"], f"output_{unique_name}.xml"
                    )
                    logger.info(f"Generating XML predictions for {unique_name} using YOLO with all predictors")
                    # Use YOLO-based prediction for toepad, detector-based for others, or original function
                    if view_type.lower() == "toepad" and yolo_model_path and os.path.exists(yolo_model_path):
                        logger.info(f"Using YOLO detection for toepad view: {yolo_model_path}")
                        utils.predictions_to_xml_single_with_yolo(
                            image_path, 
                            xml_output_path, 
                            yolo_model_path,
                            toe_predictor_path=TOEPAD_TOE_PREDICTOR,
                            scale_predictor_path=TOEPAD_SCALE_PREDICTOR,
                            finger_predictor_path=TOEPAD_FINGER_PREDICTOR,
                            target_predictor_type=toepad_predictor_type
                        )
                        logger.info(f"YOLO processing completed for {unique_name}")
                    elif detector_file_path and os.path.exists(detector_file_path):
                        utils.predictions_to_xml_single_with_detector(
                            predictor_file_path, image_path, xml_output_path, detector_file_path
                        )
                    else:
                        logger.warning(f"Not using YOLO - view_type={view_type}, yolo_model_path={yolo_model_path}, exists={os.path.exists(yolo_model_path) if yolo_model_path else False}")
                        utils.predictions_to_xml_single(
                            predictor_file_path, image_path, xml_output_path
                        )

                    # Generate CSV and TPS output files in the session outputs folder
                    csv_output_path = os.path.join(
                        session_data["outputs_folder"], f"output_{unique_name}.csv"
                    )
                    tps_output_path = os.path.join(
                        session_data["outputs_folder"], f"output_{unique_name}.tps"
                    )
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
                    data["session_id"] = session_id
                    data["view_type"] = view_type
                    all_data.append(data)

                    logger.info(
                        f"Processed image: {unique_name} for session: {session_id[:8]}"
                    )
                    
                    # Force garbage collection after each image
                    import gc
                    gc.collect()
                    
                except Exception as img_error:
                    logger.error(f"Error processing image {unique_name}: {str(img_error)}", exc_info=True)
                    # Add error info to response so client knows what went wrong
                    all_data.append({
                        "name": unique_name,
                        "error": str(img_error),
                        "coords": [],
                        "bounding_boxes": []
                    })
                    # Continue with next image instead of failing completely
                    continue

        return jsonify(all_data)

    except Exception as e:
        logger.error(f"Error in upload: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/image", methods=["POST"])
@cross_origin()
@track_metrics
def get_input_image():
    image_filename = request.args.get("image_filename")
    if not image_filename:
        return jsonify({"error": "image_filename query parameter is required"}), 400

    try:
        # Get session-specific folders
        session_id = get_session_id()
        session_data = get_session_folders(session_id)

        image_data = {}

        # Define exact paths for each image version using session folders
        paths = {
            "image1": os.path.join(
                session_data["processed_folder"], f"processed_{image_filename}"
            ),
            "image2": os.path.join(
                session_data["inverted_folder"], f"inverted_{image_filename}"
            ),
            "image3": os.path.join(session_data["upload_folder"], image_filename),
        }

        for key, path in paths.items():
            if not os.path.exists(path):
                logger.warning(f"Image not found at {path}")
                # Don't fail - continue with other images
                image_data[key] = ""
                continue

            # Read file in chunks to reduce memory usage
            try:
                with open(path, "rb") as f:
                    # Read in chunks to avoid loading entire file into memory at once
                    chunk_size = 8192  # 8KB chunks
                    chunks = []
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    
                    # Combine chunks and encode
                    file_data = b''.join(chunks)
                    image_data[key] = b64encode(file_data).decode("utf-8")
                    
                    # Clear chunks to free memory
                    del chunks
                    del file_data
                    
            except Exception as read_error:
                logger.error(f"Error reading image {path}: {str(read_error)}")
                image_data[key] = ""

        # Check if we have at least the original image
        if not image_data["image3"]:
            return jsonify({"error": "Original image not found"}), 404

        # Force garbage collection after processing
        import gc
        gc.collect()

        return jsonify(image_data)

    except Exception as e:
        logger.error(f"Error getting image: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/endpoint", methods=["POST"])
@cross_origin()
@track_metrics
def process_scatter_data():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    coords = data.get("coords")
    name = data.get("name")
    view_type = data.get("view_type", "dorsal")

    if not coords or not name:
        return jsonify({"error": "Missing required data: coords or name"}), 400

    try:
        # Get view type configuration
        predictor_file_path, detector_file_path, yolo_model_path = get_view_type_config(view_type)
        
        # Get session-specific folders
        session_id = get_session_id()
        session_data = get_session_folders(session_id)

        # Remove file extension if present
        base_name = name.split(".")[0] if "." in name else name

        # Create export directory for this output
        export_dir = export_handler.create_export_directory(name)

        # Create TPS file in session TPS folder and export directory
        tps_file_path = os.path.join(session_data["tps_folder"], f"{base_name}.tps")
        export_tps_path = export_handler.export_tps_file(coords, name, export_dir)

        logger.info(f"Creating TPS file at: {tps_file_path}")
        logger.info(f"Creating TPS file in export directory: {export_tps_path}")
        logger.info(f"Number of coordinates: {len(coords)}")
        logger.info(f"First coordinate: {coords[0] if coords else 'No coordinates'}")

        with open(tps_file_path, "w", encoding="utf-8", newline="\n") as tps_file:
            tps_file.write(f"LM={len(coords)}\n")
            # Get image height for y-flip
            from PIL import Image

            image_path = os.path.join(session_data["upload_folder"], name)
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
                tps_file_path, session_data["image_download_folder"]
            )

            # Copy annotated images to export directory
            for path in output_paths:
                export_handler.copy_file_to_export(path, export_dir)

            # Construct URLs for the generated images
            image_urls = []
            if output_paths:
                for path in output_paths:
                    image_urls.append(
                        f"/api/images/{session_id[:8]}/{os.path.basename(path)}"
                    )

            logger.info(f"Annotated images created: {output_paths}")

            return jsonify(
                {
                    "message": "TPS file and annotated image generated successfully",
                    "tps_file": tps_file_path,
                    "export_dir": export_dir,
                    "image_paths": output_paths,
                    "image_urls": image_urls,
                    "session_id": session_id,
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
                    "session_id": session_id,
                }
            )
    except Exception as e:
        logger.error(f"Error processing scatter data: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
        logger.error(f"Error creating annotated image: {str(e)}", exc_info=True)
        return jsonify(
            {
                "message": "TPS file generated but image creation failed",
                "tps_file": tps_file_path,
                "export_dir": export_dir,
                "error": str(e),
            }
        )


@app.route("/images/<session_id_short>/<path:filename>")
@track_metrics
def serve_session_image(session_id_short, filename):
    """Serve images from session-specific folders."""
    try:
        # Find session by short ID
        sessions = session_manager.list_sessions()
        session_data = None

        for session in sessions:
            if session["session_id_short"] == session_id_short:
                session_folder = session["session_folder"]
                image_download_folder = os.path.join(session_folder, "annotated")
                if os.path.exists(os.path.join(image_download_folder, filename)):
                    return send_from_directory(image_download_folder, filename)
                break

        # Fallback to global folder for backward compatibility
        return send_from_directory(IMAGE_DOWNLOAD_FOLDER, filename)

    except Exception as e:
        logger.error(f"Error serving session image: {str(e)}", exc_info=True)
        return jsonify({"error": "Image not found"}), 404


@app.route("/images/<path:filename>")
@track_metrics
def serve_image(filename):
    """Serve images from the current session or global folder."""
    try:
        # Try to get current session's image folder
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            session_data = session_manager.get_session(session_id)
            if session_data:
                image_download_folder = session_data["image_download_folder"]
                image_path = os.path.join(image_download_folder, filename)
                if os.path.exists(image_path):
                    return send_from_directory(image_download_folder, filename)

        # Fallback to global folder
        return send_from_directory(IMAGE_DOWNLOAD_FOLDER, filename)

    except Exception as e:
        logger.error(f"Error serving image: {str(e)}", exc_info=True)
        return jsonify({"error": "Image not found"}), 404


# New endpoint to list all files in the session upload folder
@app.route("/list_uploads", methods=["GET"])
@cross_origin()
@track_metrics
def list_uploads():
    try:
        session_id = get_session_id()
        session_data = get_session_folders(session_id)
        upload_folder = session_data["upload_folder"]

        if not os.path.exists(upload_folder):
            return jsonify([]), 200

        # Get all files from the session upload folder with common image extensions
        valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}
        files = []

        for filename in os.listdir(upload_folder):
            if os.path.isfile(os.path.join(upload_folder, filename)):
                # Check if the file has a valid image extension
                _, ext = os.path.splitext(filename)
                if ext.lower() in valid_extensions:
                    files.append(filename)

        logger.info(
            f"Found {len(files)} files in session upload folder for session: {session_id[:8]}"
        )
        return jsonify(files), 200
    except Exception as e:
        logger.error(f"Error listing session upload directory: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# Endpoint to process an existing image from the session uploads folder
@app.route("/process_existing", methods=["POST"])
@cross_origin()
@track_metrics
def process_existing():
    try:
        filename = request.args.get("filename")
        if not filename:
            return jsonify({"error": "filename parameter is required"}), 400

        # Get view type from query parameters
        view_type = request.args.get("view_type", "dorsal")
        toepad_predictor_type = request.args.get("toepad_predictor_type", "toe")
        predictor_file_path, detector_file_path, yolo_model_path = get_view_type_config(view_type)
        
        # For toepad view type, select the appropriate predictor
        if view_type.lower() == "toepad":
            if toepad_predictor_type == "scale":
                predictor_file_path = TOEPAD_SCALE_PREDICTOR
            elif toepad_predictor_type == "finger":
                predictor_file_path = TOEPAD_FINGER_PREDICTOR
            else:  # default to toe
                predictor_file_path = TOEPAD_TOE_PREDICTOR
        
        # Get session-specific folders
        session_id = get_session_id()
        session_data = get_session_folders(session_id)

        image_path = os.path.join(session_data["upload_folder"], filename)
        if not os.path.exists(image_path):
            return (
                jsonify(
                    {"error": f"File {filename} not found in session uploads folder"}
                ),
                404,
            )

        # Create export directory for this processing
        export_dir = export_handler.create_export_directory(filename)

        # Check if the processed versions already exist in session folders
        processed_path = os.path.join(
            session_data["processed_folder"], f"processed_{filename}"
        )
        inverted_path = os.path.join(
            session_data["inverted_folder"], f"inverted_{filename}"
        )
        xml_path = os.path.join(
            session_data["outputs_folder"], f"output_{filename}.xml"
        )

        # Process the image if needed
        if not os.path.exists(processed_path):
            xray_preprocessing.process_single_image(image_path, processed_path)

        if not os.path.exists(inverted_path):
            visual_individual_performance.invert_single_image(image_path, inverted_path)

        # Generate XML if it doesn't exist
        if not os.path.exists(xml_path):
            # Use YOLO-based prediction for toepad, detector-based for others, or original function
            if view_type.lower() == "toepad" and yolo_model_path and os.path.exists(yolo_model_path):
                utils.predictions_to_xml_single_with_yolo(
                    image_path, 
                    xml_path, 
                    yolo_model_path,
                    toe_predictor_path=TOEPAD_TOE_PREDICTOR,
                    scale_predictor_path=TOEPAD_SCALE_PREDICTOR,
                    finger_predictor_path=TOEPAD_FINGER_PREDICTOR,
                    target_predictor_type=toepad_predictor_type
                )
            elif detector_file_path and os.path.exists(detector_file_path):
                utils.predictions_to_xml_single_with_detector(
                    predictor_file_path, image_path, xml_path, detector_file_path
                )
            else:
                utils.predictions_to_xml_single(
                    predictor_file_path, image_path, xml_path
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
        csv_path = os.path.join(
            session_data["outputs_folder"], f"output_{filename}.csv"
        )
        tps_path = os.path.join(
            session_data["outputs_folder"], f"output_{filename}.tps"
        )
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
        data["session_id"] = session_id
        data["view_type"] = view_type

        logger.info(
            f"Processed existing image: {filename} for session: {session_id[:8]} with view type: {view_type}"
        )
        return jsonify(data)

    except Exception as e:
        logger.error(f"Error processing existing image: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# New endpoint to save annotations (updated landmark points)
@app.route("/save_annotations", methods=["POST"])
@cross_origin()
@track_metrics
def save_annotations():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        coords = data.get("coords")
        name = data.get("name")
        view_type = data.get("view_type", "dorsal")

        if not coords or not name:
            return jsonify({"error": "Missing required data: coords or name"}), 400

        # Get view type configuration
        predictor_file_path, detector_file_path, yolo_model_path = get_view_type_config(view_type)
        
        # For toepad view type, select the appropriate predictor (default to toe)
        if view_type.lower() == "toepad":
            toepad_predictor_type = data.get("toepad_predictor_type", "toe")
            if toepad_predictor_type == "scale":
                predictor_file_path = TOEPAD_SCALE_PREDICTOR
            elif toepad_predictor_type == "finger":
                predictor_file_path = TOEPAD_FINGER_PREDICTOR
            else:
                predictor_file_path = TOEPAD_TOE_PREDICTOR

        # Get session-specific folders
        session_id = get_session_id()
        session_data = get_session_folders(session_id)

        # Remove file extension if present for base name
        base_name = name.split(".")[0] if "." in name else name

        # Create timestamp for version control
        timestamp = int(time.time())

        # Define filenames for the output files in session folders
        xml_path = os.path.join(session_data["outputs_folder"], f"output_{name}.xml")
        csv_path = os.path.join(session_data["outputs_folder"], f"output_{name}.csv")
        tps_path = os.path.join(session_data["outputs_folder"], f"output_{name}.tps")

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
                                    x_coord = float(point["x"])
                                    y_coord = float(point["y"])
                                    part.set("x", str(int(x_coord)))
                                    part.set("y", str(int(y_coord)))

                # Save the updated XML
                utils.pretty_xml(root, xml_path)

                # Now regenerate CSV and TPS files
                utils.dlib_xml_to_pandas(xml_path)
                utils.dlib_xml_to_tps(xml_path)

                logger.info(f"Updated XML, CSV, and TPS files for {name}")
            except Exception as xml_e:
                logger.error(f"Error updating XML file: {str(xml_e)}", exc_info=True)
                # If XML update fails, create TPS file directly
                logger.info("Falling back to direct TPS file creation")

        # Create TPS file directly from coordinates
        tps_file_path = os.path.join(session_data["tps_folder"], f"{base_name}.tps")
        export_tps_path = export_handler.export_tps_file(coords, name, export_dir)

        logger.info(f"Created TPS file at: {tps_file_path}")
        logger.info(f"Created TPS file in export directory: {export_tps_path}")

        # Generate annotated image with updated points
        try:
            output_paths = visual_individual_performance.create_image(
                tps_file_path, session_data["image_download_folder"]
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

            logger.info(
                f"Annotations saved successfully for {name} in session {session_id[:8]}"
            )

            return jsonify(
                {
                    "message": "Annotations saved successfully",
                    "export_dir": export_dir,
                    "session_id": session_id,
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
                    "session_id": session_id,
                    "error": str(img_e),
                }
            )

    except Exception as e:
        logger.error(f"Error saving annotations: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

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
@track_metrics
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


@app.route("/system/status", methods=["GET"])
@cross_origin()
@track_metrics
@localhost_only
def system_status():
    """Get current system status including CPU, memory, and disk usage"""
    try:
        system_info = get_system_info()
        return jsonify({
            "success": True,
            "timestamp": time.time(),
            "system": system_info
        }), 200
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Failed to get system status: {str(e)}"
        }), 500

@app.route("/system/cpu", methods=["GET"])
@cross_origin()
@track_metrics
@localhost_only
def cpu_usage():
    """Get current CPU usage"""
    try:
        cpu_percent = get_cpu_usage()
        return jsonify({
            "success": True,
            "cpu_percent": cpu_percent,
            "timestamp": time.time()
        }), 200
    except Exception as e:
        logger.error(f"Error getting CPU usage: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Failed to get CPU usage: {str(e)}"
        }), 500

@app.route("/system/memory", methods=["GET"])
@cross_origin()
@track_metrics
@localhost_only
def memory_usage():
    """Get current memory usage"""
    try:
        memory_percent = get_memory_usage()
        memory = psutil.virtual_memory()
        return jsonify({
            "success": True,
            "memory_percent": memory_percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "timestamp": time.time()
        }), 200
    except Exception as e:
        logger.error(f"Error getting memory usage: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Failed to get memory usage: {str(e)}"
        }), 500

@app.route("/system/memory/cleanup", methods=["POST"])
@cross_origin()
@track_metrics
@localhost_only
def memory_cleanup():
    """Force memory cleanup"""
    try:
        freed_mb = cleanup_memory()
        memory = psutil.virtual_memory()
        return jsonify({
            "success": True,
            "freed_mb": freed_mb,
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "timestamp": time.time()
        }), 200
    except Exception as e:
        logger.error(f"Error during memory cleanup: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Failed to cleanup memory: {str(e)}"
        }), 500

@app.route("/clear_history", methods=["POST"])
@cross_origin()
@track_metrics
def clear_history():
    """
    Clear all files and history data for the current session.
    This includes uploaded images, processed files, outputs, and export directories for this session only.
    """
    try:
        session_id = get_session_id()

        # Use session manager to clear session-specific files
        result = session_manager.clear_session(session_id)

        if result["success"]:
            logger.info(f"Session history cleared for session: {session_id[:8]}")
            
            # Force memory cleanup after clearing history
            freed_mb = cleanup_memory()
            logger.info(f"Memory cleanup after history clear freed {freed_mb:.2f} MB")
            
            # Add memory cleanup info to response
            result["memory_cleanup"] = {
                "freed_mb": freed_mb,
                "memory_percent": get_memory_usage()
            }
            
            return jsonify(result), 200
        else:
            logger.error(
                f"Failed to clear session history: {result.get('error', 'Unknown error')}"
            )
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Error clearing session history: {str(e)}", exc_info=True)
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Failed to clear session history: {str(e)}",
                }
            ),
            500,
        )


def verify_webhook_signature(payload, signature):
    """Verify GitHub webhook signature"""
    if not signature or not WEBHOOK_SECRET or WEBHOOK_SECRET == "your-webhook-secret-here":
        return False
    
    expected_signature = 'sha256=' + hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)


@app.route("/webhook", methods=["POST"])
@cross_origin()
@track_metrics
def github_webhook():
    """GitHub webhook endpoint for auto-deployment"""
    try:
        # Get the raw payload
        payload = request.get_data()
        
        # Verify webhook signature if enabled
        if VERIFY_SIGNATURE:
            signature = request.headers.get('X-Hub-Signature-256', '')
            if not verify_webhook_signature(payload, signature):
                logger.warning("Invalid webhook signature")
                return jsonify({"error": "Invalid signature"}), 401
        
        # Parse JSON payload
        data = request.get_json()
        
        # Check if this is a push to main branch
        if (data.get('ref') == f'refs/heads/{MAIN_BRANCH}' and 
            data.get('repository', {}).get('name') == REPO_NAME):
            
            logger.info(f"Push to {MAIN_BRANCH} branch detected, triggering deployment...")
            
            # Trigger deployment in a separate thread to avoid blocking
            def deploy_async():
                try:
                    logger.info("Starting deployment process...")
                    
                    # Change to repository directory
                    repo_path = "/var/www/LizardMorph"
                    os.chdir(repo_path)
                    
                    # Stash any uncommitted changes
                    if subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"], capture_output=True).returncode != 0:
                        logger.info("Stashing uncommitted changes...")
                        subprocess.run(["git", "stash", "push", "-m", f"Auto-deploy stash {time.time()}"], check=True)
                    
                    # Fetch latest changes
                    logger.info("Fetching latest changes...")
                    subprocess.run(["git", "fetch", "origin"], check=True)
                    
                    # Check if main has new commits
                    current_branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
                    main_commits = subprocess.run(["git", "log", "--oneline", f"{current_branch}..origin/main"], capture_output=True, text=True).stdout.strip()
                    
                    if not main_commits:
                        logger.info("No new commits on main branch")
                        return
                    
                    logger.info(f"Found new commits on main branch")
                    
                    # Pull latest changes
                    logger.info("Pulling latest changes from main branch...")
                    subprocess.run(["git", "pull", "origin", "main"], check=True)
                    logger.info("Successfully updated to latest main")
                    
                    # Rebuild frontend if it exists
                    frontend_path = os.path.join(repo_path, "frontend")
                    if os.path.exists(frontend_path) and os.path.exists(os.path.join(frontend_path, "package.json")):
                        logger.info("Rebuilding frontend...")
                        os.chdir(frontend_path)
                        subprocess.run(["npm", "install", "--production"], check=True)
                        subprocess.run(["npm", "run", "build"], check=True)
                        logger.info("Frontend build completed")
                        os.chdir(repo_path)
                    
                    # Restart backend service
                    logger.info("Restarting backend service...")
                    try:
                        # Try systemctl first (if service exists)
                        subprocess.run(["systemctl", "restart", "lizardmorph-backend"], check=True)
                        logger.info("Backend service restarted via systemctl")
                    except subprocess.CalledProcessError:
                        # Fallback: restart the process manually
                        logger.info("Systemctl failed, restarting backend manually...")
                        # Kill existing gunicorn processes
                        subprocess.run(["pkill", "-f", "gunicorn.*app:app"], check=False)
                        # Start new backend process
                        backend_dir = os.path.join(repo_path, "backend")
                        os.chdir(backend_dir)
                        subprocess.Popen([
                            "gunicorn", "-c", "../gunicorn.conf.py", "app:app"
                        ], cwd=backend_dir)
                        logger.info("Backend service restarted manually")
                        os.chdir(repo_path)
                    
                    # Reload nginx
                    logger.info("Reloading nginx...")
                    subprocess.run(["nginx", "-s", "reload"], check=True)
                    logger.info("Nginx reloaded")
                    
                    logger.info("Deployment completed successfully!")
                    
                except subprocess.CalledProcessError as e:
                    logger.error(f"Deployment failed: {e}")
                except Exception as e:
                    logger.error(f"Deployment error: {str(e)}")
            
            # Start deployment in background thread
            thread = threading.Thread(target=deploy_async)
            thread.daemon = True
            thread.start()
            
            return jsonify({"status": "success", "message": "Deployment triggered"}), 200
            
        else:
            logger.info(f"Push event received but not to {MAIN_BRANCH} branch, ignoring")
            return jsonify({"status": "ignored", "message": "Not a push to main branch"}), 200
            
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
        return jsonify({"error": f"Webhook processing error: {str(e)}"}), 500


# Make sure your app runs on the correct host and port if started directly
if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 5000))
    app.run(host="0.0.0.0", port=port)
