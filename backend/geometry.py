from typing import Sequence, List


def canonicalize_obb_for_crop(obb: Sequence[float]) -> List[float]:
    """Return one deterministic representation of an equivalent OBB crop.

    A rectangle can be represented as ``(w, h, angle)`` or as
    ``(h, w, angle - 90)``. YOLO-OBB commonly returns the latter even when the
    source annotation used the former. Landmark predictors need a stable crop
    orientation, so always make the long side the crop width and normalize the
    angle to [-90, 90).
    """
    if len(obb) < 5:
        raise ValueError("An oriented bounding box requires five values.")

    cx, cy, width, height, angle = (float(value) for value in obb[:5])
    if width < height:
        width, height = height, width
        angle -= 90.0

    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0

    return [cx, cy, width, height, angle]
