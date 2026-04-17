"""
Platform-native OCR: Apple Vision (macOS), WinRT (Windows), or Tesseract (Linux).
Drop-in replacement for easyocr in id_extractor.py — avoids torch when possible.
"""

import logging
import platform
import shutil
import numpy as np
import cv2
import re

_logger = logging.getLogger(__name__)


def _create_reader():
    """Create a platform-native OCR reader."""
    system = platform.system()
    if system == "Darwin":
        return _MacOSReader()
    elif system == "Windows":
        return _WindowsReader()
    elif system == "Linux":
        try:
            return _LinuxTesseractReader()
        except ImportError:
            _logger.warning(
                "Linux: tesseract not available (install apt package tesseract-ocr). "
                "ID OCR will return no text until it is installed."
            )
            return _LinuxTesseractPlaceholderReader()
    else:
        raise ImportError(f"No native OCR available for {system}. Install easyocr as fallback.")


class _LinuxTesseractPlaceholderReader:
    """No-op OCR when tesseract is missing; avoids loading EasyOCR/torch on small servers."""

    def readtext(self, image, detail=1, allowlist=None):
        return []


class _MacOSReader:
    """OCR using Apple Vision framework via pyobjc."""

    def __init__(self):
        import Vision  # noqa: F401 — verify availability

    def readtext(self, image, detail=1, allowlist=None):
        """
        Match easyocr's readtext interface.
        Returns list of (bbox, text, confidence).
        """
        import Vision
        import Quartz

        # Convert numpy array to CGImage
        if len(image.shape) == 2:
            # Grayscale — convert to RGB for Vision
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w, c = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bytes_per_row = w * c
        color_space = Quartz.CGColorSpaceCreateDeviceRGB()
        data_provider = Quartz.CGDataProviderCreateWithData(
            None, rgb.tobytes(), h * bytes_per_row, None
        )
        cg_image = Quartz.CGImageCreate(
            w, h, 8, 8 * c, bytes_per_row, color_space,
            Quartz.kCGImageAlphaNone,
            data_provider, None, False, Quartz.kCGRenderingIntentDefault,
        )

        # Create and run VNRecognizeTextRequest
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)

        if allowlist:
            request.setCustomWords_([allowlist])

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        success = handler.performRequests_error_([request], None)

        results = []
        if success and request.results():
            for obs in request.results():
                candidate = obs.topCandidates_(1)[0]
                text = candidate.string()
                conf = candidate.confidence()

                # Filter to allowlist if specified
                if allowlist:
                    text = re.sub(f"[^{re.escape(allowlist)}]", "", text)

                if not text:
                    continue

                # Get bounding box (Vision uses bottom-left origin, normalized)
                box = obs.boundingBox()
                x = box.origin.x * w
                y = (1 - box.origin.y - box.size.height) * h
                bw = box.size.width * w
                bh = box.size.height * h
                bbox = [
                    [x, y],
                    [x + bw, y],
                    [x + bw, y + bh],
                    [x, y + bh],
                ]

                results.append((bbox, text, conf))

        return results


class _WindowsReader:
    """OCR using Windows.Media.Ocr (WinRT)."""

    def __init__(self):
        try:
            import winocr  # noqa: F401
        except ImportError:
            raise ImportError("winocr not installed. Run: pip install winocr")

    def readtext(self, image, detail=1, allowlist=None):
        """
        Match easyocr's readtext interface.
        Returns list of (bbox, text, confidence).
        """
        import winocr
        import asyncio

        # Encode image to PNG bytes for winocr
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        _, png_bytes = cv2.imencode(".png", image)

        # Run async OCR
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                winocr.recognize_cv2(image, lang="en")
            )
        finally:
            loop.close()

        h, w = image.shape[:2]
        results = []
        for line in result["lines"]:
            text = line["text"]
            # WinRT doesn't provide per-line confidence, estimate 0.9
            conf = 0.9

            if allowlist:
                text = re.sub(f"[^{re.escape(allowlist)}]", "", text)

            if not text:
                continue

            # winocr returns bounding box as dict with x, y, width, height
            bx = line.get("x", 0)
            by = line.get("y", 0)
            bw = line.get("width", w)
            bh = line.get("height", h)
            bbox = [
                [bx, by],
                [bx + bw, by],
                [bx + bw, by + bh],
                [bx, by + bh],
            ]

            results.append((bbox, text, conf))

        return results


class _LinuxTesseractReader:
    """OCR using the tesseract binary (no torch). Requires `tesseract` on PATH."""

    def __init__(self):
        if not shutil.which("tesseract"):
            raise ImportError("tesseract binary not found; install e.g. apt install tesseract-ocr")
        import pytesseract

        pytesseract.get_tesseract_version()

    def readtext(self, image, detail=1, allowlist=None):
        """Match easyocr readtext: list of (bbox, text, confidence)."""
        import pytesseract
        from pytesseract import Output

        if image is None or image.size == 0:
            return []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        config = "--oem 3 --psm 7"
        if allowlist:
            config += f" -c tessedit_char_whitelist={allowlist}"

        text = pytesseract.image_to_string(gray, config=config).strip()
        if not text:
            return []

        data = pytesseract.image_to_data(gray, config=config, output_type=Output.DICT)
        confs = []
        for i, w in enumerate(data.get("text", [])):
            w = (w or "").strip()
            if not w:
                continue
            try:
                c = int(data["conf"][i])
            except (ValueError, IndexError, KeyError):
                continue
            if c >= 0:
                confs.append(c / 100.0)
        conf = sum(confs) / len(confs) if confs else 0.75

        h, w = gray.shape[:2]
        bbox = [[0, 0], [w, 0], [w, h], [0, h]]
        return [(bbox, text, float(conf))]
