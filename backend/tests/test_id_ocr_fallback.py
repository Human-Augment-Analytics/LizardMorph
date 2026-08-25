import builtins

import numpy as np
import pytest

import id_extractor
import native_ocr


@pytest.fixture()
def without_native_ocr(monkeypatch):
    def unavailable():
        raise ImportError("Vision is not bundled")

    monkeypatch.setattr(id_extractor, "_reader", None, raising=False)
    monkeypatch.setattr(native_ocr, "_create_reader", unavailable)


@pytest.fixture()
def without_easyocr(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "easyocr":
            raise ImportError("No module named 'easyocr'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


def test_ocr_degrades_to_no_text_when_no_backend_is_bundled(without_native_ocr, without_easyocr):
    text, confidence = id_extractor.detect_digits(np.zeros((64, 128, 3), dtype=np.uint8))

    assert text == ""
    assert confidence == 0.0


def test_extract_id_reports_an_empty_id_rather_than_an_import_error(
    without_native_ocr, without_easyocr, tmp_path
):
    import cv2

    image_path = tmp_path / "toepad.png"
    cv2.imwrite(str(image_path), np.zeros((200, 200, 3), dtype=np.uint8))

    result = id_extractor.extract_id_from_image(str(image_path), (0.5, 0.5, 0.4, 0.4))

    assert "error" not in result
    assert result["id"] == ""


def test_ocr_backend_is_resolved_once_and_cached(without_native_ocr, without_easyocr):
    first = id_extractor._get_reader()
    second = id_extractor._get_reader()

    assert first is second
    assert first.readtext(np.zeros((8, 8, 3), dtype=np.uint8)) == []
