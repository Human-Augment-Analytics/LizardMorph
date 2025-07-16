import os
import shutil
import uuid
import time
import logging
from datetime import datetime
from typing import Optional, Dict, List

# Configure logging
logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages user sessions by creating isolated folders for each session
    and handling cleanup of session-specific files.
    """

    def __init__(self, base_sessions_dir="sessions"):
        """
        Initialize the session manager.

        Args:
            base_sessions_dir (str): Base directory for all session folders
        """
        self.base_sessions_dir = base_sessions_dir
        self.active_sessions = {}
        self.ensure_directory_exists(self.base_sessions_dir)

    def ensure_directory_exists(self, directory):
        """Create directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

    def create_session(self, session_id: str = None) -> str:
        """
        Create a new session with a unique session ID.

        Returns:
            str: Unique session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        else:
            session_id = session_id

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        session_data = {
            "session_id": session_id,
            "created_at": timestamp,
            "session_folder": os.path.join(
                self.base_sessions_dir, f"session_{timestamp}_{session_id[:8]}"
            ),
            "upload_folder": None,
            "processed_folder": None,
            "inverted_folder": None,
            "tps_folder": None,
            "image_download_folder": None,
            "outputs_folder": None,
        }

        # Create session folder and subfolders
        self.ensure_directory_exists(session_data["session_folder"])

        # Create session-specific subfolders
        session_data["upload_folder"] = os.path.join(
            session_data["session_folder"], "uploads"
        )
        session_data["processed_folder"] = os.path.join(
            session_data["session_folder"], "processed"
        )
        session_data["inverted_folder"] = os.path.join(
            session_data["session_folder"], "inverted"
        )
        session_data["tps_folder"] = os.path.join(session_data["session_folder"], "tps")
        session_data["image_download_folder"] = os.path.join(
            session_data["session_folder"], "annotated"
        )
        session_data["outputs_folder"] = os.path.join(
            session_data["session_folder"], "outputs"
        )

        for folder in [
            session_data["upload_folder"],
            session_data["processed_folder"],
            session_data["inverted_folder"],
            session_data["tps_folder"],
            session_data["image_download_folder"],
            session_data["outputs_folder"],
        ]:
            self.ensure_directory_exists(folder)

        self.active_sessions[session_id] = session_data
        logger.info(
            f"Created new session: {session_id} at {session_data['session_folder']}"
        )

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get session data by session ID.

        Args:
            session_id (str): Session ID

        Returns:
            dict: Session data or None if not found
        """
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]

        # Try to load session from filesystem if not in memory
        return self._load_session_from_filesystem(session_id)

    def _load_session_from_filesystem(self, session_id: str) -> Optional[Dict]:
        """
        Load session data from filesystem if it exists.

        Args:
            session_id (str): Session ID

        Returns:
            dict: Session data or None if not found
        """
        # Look for session folders that contain this session ID
        if not os.path.exists(self.base_sessions_dir):
            return None

        for folder_name in os.listdir(self.base_sessions_dir):
            if session_id[:8] in folder_name and folder_name.startswith("session_"):
                session_folder = os.path.join(self.base_sessions_dir, folder_name)
                if os.path.isdir(session_folder):
                    # Reconstruct session data
                    session_data = {
                        "session_id": session_id,
                        "created_at": folder_name.split("_")[1],
                        "session_folder": session_folder,
                        "upload_folder": os.path.join(session_folder, "uploads"),
                        "processed_folder": os.path.join(session_folder, "processed"),
                        "inverted_folder": os.path.join(session_folder, "inverted"),
                        "tps_folder": os.path.join(session_folder, "tps"),
                        "image_download_folder": os.path.join(
                            session_folder, "annotated"
                        ),
                        "outputs_folder": os.path.join(session_folder, "outputs"),
                    }

                    # Ensure all subfolders exist
                    for folder in [
                        session_data["upload_folder"],
                        session_data["processed_folder"],
                        session_data["inverted_folder"],
                        session_data["tps_folder"],
                        session_data["image_download_folder"],
                        session_data["outputs_folder"],
                    ]:
                        self.ensure_directory_exists(folder)

                    self.active_sessions[session_id] = session_data
                    logger.info(f"Loaded existing session: {session_id}")
                    return session_data

        return None

    def clear_session(self, session_id: str) -> Dict:
        """
        Clear all files for a specific session.

        Args:
            session_id (str): Session ID to clear

        Returns:
            dict: Results of the clearing operation
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return {"success": False, "error": f"Session {session_id} not found"}

        cleared_items = []
        errors = []

        # Clear session folder contents
        session_folder = session_data["session_folder"]

        try:
            if os.path.exists(session_folder):
                item_count = 0
                for root, dirs, files in os.walk(session_folder):
                    item_count += len(files)

                if item_count > 0:
                    # Clear all files in the session folder but keep the folder structure
                    for root, dirs, files in os.walk(session_folder):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                os.unlink(file_path)
                            except Exception as file_error:
                                logger.warning(
                                    f"Could not delete {file_path}: {str(file_error)}"
                                )
                                errors.append(f"Could not delete {file}")

                    cleared_items.append(
                        f"{item_count} files from session {session_id[:8]}"
                    )
                    logger.info(f"Cleared {item_count} files from session {session_id}")
                else:
                    cleared_items.append(f"Session {session_id[:8]} was already empty")
            else:
                cleared_items.append(f"Session {session_id[:8]} folder not found")

        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {str(e)}")
            errors.append(f"Error clearing session: {str(e)}")

        # Clear any output files in the backend root directory that might be related to this session
        try:
            backend_dir = os.getcwd()
            root_files_cleared = 0
            for filename in os.listdir(backend_dir):
                if filename.startswith("output_") and (
                    filename.endswith(".xml")
                    or filename.endswith(".csv")
                    or filename.endswith(".tps")
                    or filename.endswith(".bak")
                ):
                    try:
                        file_path = os.path.join(backend_dir, filename)
                        os.unlink(file_path)
                        root_files_cleared += 1
                    except Exception as file_error:
                        logger.warning(
                            f"Could not delete {filename}: {str(file_error)}"
                        )
                        errors.append(f"Could not delete {filename}")

            if root_files_cleared > 0:
                cleared_items.append(
                    f"{root_files_cleared} output files from backend root"
                )

        except Exception as root_error:
            logger.error(f"Error clearing root directory files: {str(root_error)}")
            errors.append(f"Error clearing root directory files: {str(root_error)}")

        result = {
            "success": True,
            "session_id": session_id,
            "cleared_items": cleared_items,
        }

        if errors:
            result["warnings"] = errors

        return result

    def list_sessions(self) -> List[Dict]:
        """
        List all available sessions.

        Returns:
            list: List of session information
        """
        sessions = []

        if not os.path.exists(self.base_sessions_dir):
            return sessions

        for folder_name in os.listdir(self.base_sessions_dir):
            if folder_name.startswith("session_") and os.path.isdir(
                os.path.join(self.base_sessions_dir, folder_name)
            ):
                parts = folder_name.split("_")
                if len(parts) >= 3:
                    timestamp = parts[1]
                    session_id_part = parts[2]

                    session_folder = os.path.join(self.base_sessions_dir, folder_name)

                    # Count files in session
                    file_count = 0
                    for root, dirs, files in os.walk(session_folder):
                        file_count += len(files)

                    sessions.append(
                        {
                            "session_id_short": session_id_part,
                            "created_at": timestamp,
                            "folder_name": folder_name,
                            "session_folder": session_folder,
                            "file_count": file_count,
                        }
                    )

        # Sort by creation time, newest first
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> Dict:
        """
        Completely delete a session and its folder.

        Args:
            session_id (str): Session ID to delete

        Returns:
            dict: Results of the deletion operation
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return {"success": False, "error": f"Session {session_id} not found"}

        session_folder = session_data["session_folder"]

        try:
            if os.path.exists(session_folder):
                shutil.rmtree(session_folder)
                logger.info(f"Deleted session folder: {session_folder}")

                # Remove from active sessions
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]

                return {
                    "success": True,
                    "message": f"Session {session_id[:8]} deleted successfully",
                }
            else:
                return {
                    "success": False,
                    "error": f"Session folder not found: {session_folder}",
                }

        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {str(e)}")
            return {"success": False, "error": f"Error deleting session: {str(e)}"}
