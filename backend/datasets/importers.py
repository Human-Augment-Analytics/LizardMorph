import os
import math
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple, Any
try:
    from backend.datasets.canonical import (
        LandmarkPoint,
        CanonicalObject,
        CanonicalImage,
        CanonicalDataset,
    )
except ImportError:
    from datasets.canonical import (
        LandmarkPoint,
        CanonicalObject,
        CanonicalImage,
        CanonicalDataset,
    )


class TPSImporter:
    """Importer for Thin-Plate Spline (TPS) landmark dataset files."""

    @classmethod
    def parse_string(cls, content: str, default_image_name: str = "image.jpg") -> CanonicalDataset:
        lines = content.strip().splitlines()
        images_dict: Dict[str, List[CanonicalObject]] = {}
        
        current_lm_count: Optional[int] = None
        current_pts: List[LandmarkPoint] = []
        current_tags: Dict[str, str] = {}
        obj_counter = 0

        def finalize_object():
            nonlocal obj_counter, current_pts, current_tags, current_lm_count
            if current_lm_count is None and not current_pts:
                return
            if current_lm_count is None or current_lm_count <= 0:
                raise ValueError("TPS specimens must declare at least one landmark.")
            if len(current_pts) != current_lm_count:
                raise ValueError(
                    f"TPS specimen declares {current_lm_count} landmarks but contains {len(current_pts)}."
                )

            file_path = current_tags.get("IMAGE", current_tags.get("image", default_image_name))
            specimen_id = current_tags.get("ID", current_tags.get("id"))
            
            pts_tuples = [(pt.x, pt.y) for pt in current_pts]
            obb = CanonicalDataset.derive_obb_from_landmarks(pts_tuples)

            obj_counter += 1
            obj = CanonicalObject(
                object_id=f"obj_{obj_counter}",
                class_name="object",
                obb=obb,
                landmarks=list(current_pts),
                specimen_id=specimen_id,
            )

            if file_path not in images_dict:
                images_dict[file_path] = []
            images_dict[file_path].append(obj)

            current_lm_count = None
            current_pts = []
            current_tags = {}

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            if "=" in line:
                key, val = line.split("=", 1)
                key_upper = key.strip().upper()
                val_clean = val.strip()

                if key_upper == "LM":
                    if current_lm_count is not None or current_pts:
                        finalize_object()
                    current_lm_count = int(val_clean)
                    if current_lm_count <= 0:
                        raise ValueError("TPS landmark count must be greater than zero.")
                    # Read next current_lm_count lines
                    for pt_idx in range(current_lm_count):
                        i += 1
                        if i >= len(lines):
                            raise ValueError("TPS file ended before all landmarks were read.")
                        pt_line = lines[i].strip()
                        parts = pt_line.split()
                        if len(parts) < 2:
                            raise ValueError(f"Invalid TPS landmark line: '{pt_line}'.")
                        x, y = float(parts[0]), float(parts[1])
                        current_pts.append(
                            LandmarkPoint(name=str(pt_idx), x=x, y=y)
                        )
                else:
                    current_tags[key_upper] = val_clean
            i += 1

        if current_lm_count is not None or current_pts or current_tags:
            finalize_object()

        images = []
        for img_idx, (path, objs) in enumerate(images_dict.items()):
            images.append(
                CanonicalImage(
                    image_id=f"img_{img_idx + 1}",
                    file_path=path,
                    width=0,
                    height=0,
                    objects=objs,
                )
            )

        return CanonicalDataset(images=images)

    @classmethod
    def parse_file(cls, file_path: str) -> CanonicalDataset:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.parse_string(content, default_image_name=os.path.basename(file_path))


class DlibXMLImporter:
    """Importer for dlib XML annotation dataset files."""

    @classmethod
    def parse_string(cls, content: str) -> CanonicalDataset:
        root = ET.fromstring(content)
        images = []

        images_node = root.find("images")
        image_nodes = images_node.findall("image") if images_node is not None else root.findall("image")

        for img_idx, img_elem in enumerate(image_nodes):
            file_path = img_elem.get("file", f"image_{img_idx + 1}.jpg")
            width = int(img_elem.get("width", 0))
            height = int(img_elem.get("height", 0))

            objects = []
            for box_idx, box_elem in enumerate(img_elem.findall("box")):
                label_child = box_elem.find("label")
                label_text = (
                    label_child.text.strip()
                    if label_child is not None and label_child.text
                    else None
                )
                class_name = (box_elem.get("label") or label_text or "object").strip()

                spec_child = box_elem.find("specimen_id")
                spec_text = (
                    spec_child.text.strip()
                    if spec_child is not None and spec_child.text
                    else None
                )
                specimen_id = box_elem.get("specimen_id") or spec_text
                
                parts = []
                for part_idx, part_elem in enumerate(box_elem.findall("part")):
                    p_name = part_elem.get("name", str(part_idx))
                    p_x = float(part_elem.get("x", 0.0))
                    p_y = float(part_elem.get("y", 0.0))
                    parts.append(LandmarkPoint(name=p_name, x=p_x, y=p_y))

                top = box_elem.get("top")
                left = box_elem.get("left")
                box_w = box_elem.get("width")
                box_h = box_elem.get("height")

                if top is not None and left is not None and box_w is not None and box_h is not None:
                    top_f = float(top)
                    left_f = float(left)
                    w_f = float(box_w)
                    h_f = float(box_h)
                    cx = left_f + w_f / 2.0
                    cy = top_f + h_f / 2.0
                    obb = [cx, cy, w_f, h_f, 0.0]
                elif parts:
                    pts_tuples = [(p.x, p.y) for p in parts]
                    obb = CanonicalDataset.derive_obb_from_landmarks(pts_tuples)
                else:
                    obb = [0.0, 0.0, 0.0, 0.0, 0.0]

                objects.append(
                    CanonicalObject(
                        object_id=f"obj_{img_idx + 1}_{box_idx + 1}",
                        class_name=class_name,
                        obb=obb,
                        landmarks=parts,
                        specimen_id=specimen_id,
                    )
                )

            images.append(
                CanonicalImage(
                    image_id=f"img_{img_idx + 1}",
                    file_path=file_path,
                    width=width,
                    height=height,
                    objects=objects,
                )
            )

        return CanonicalDataset(images=images)

    @classmethod
    def parse_file(cls, file_path: str) -> CanonicalDataset:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.parse_string(content)


class YOLOOBBImporter:
    """Importer for YOLO OBB text annotation dataset files."""

    @classmethod
    def parse_string(
        cls,
        content: str,
        file_path: str = "image.jpg",
        width: int = 0,
        height: int = 0,
        class_names: Optional[Dict[int, str]] = None,
    ) -> CanonicalDataset:
        if class_names is None:
            class_names = {}

        objects = []
        lines = content.strip().splitlines()

        for obj_idx, line in enumerate(lines):
            line_str = line.strip()
            if not line_str:
                continue

            parts = line_str.split()
            if len(parts) < 6:
                continue

            try:
                class_id = int(float(parts[0]))
            except ValueError:
                class_id = 0

            class_name = class_names.get(class_id, str(class_id))
            floats = [float(p) for p in parts[1:]]

            if len(floats) >= 8:
                # Format: x1 y1 x2 y2 x3 y3 x4 y4
                x1, y1, x2, y2, x3, y3, x4, y4 = floats[:8]
                
                # Check if coordinates are normalized (max <= 1.0) and scale if dimensions provided
                max_val = max(abs(x1), abs(y1), abs(x2), abs(y2), abs(x3), abs(y3), abs(x4), abs(y4))
                if max_val <= 1.0 and width > 0 and height > 0:
                    x1 *= width
                    x3 *= width
                    x2 *= width
                    x4 *= width
                    y1 *= height
                    y2 *= height
                    y3 *= height
                    y4 *= height

                cx = (x1 + x2 + x3 + x4) / 4.0
                cy = (y1 + y2 + y3 + y4) / 4.0
                w = math.hypot(x2 - x1, y2 - y1)
                h = math.hypot(x3 - x2, y3 - y2)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                obb = [cx, cy, w, h, angle]
            elif len(floats) >= 5:
                # Format: cx cy w h angle
                cx, cy, w, h, angle = floats[:5]
                max_val = max(abs(cx), abs(cy), abs(w), abs(h))
                if max_val <= 1.0 and width > 0 and height > 0:
                    cx *= width
                    cy *= height
                    w *= width
                    h *= height
                obb = [cx, cy, w, h, angle]
            else:
                continue

            objects.append(
                CanonicalObject(
                    object_id=f"obj_1_{obj_idx + 1}",
                    class_name=class_name,
                    obb=obb,
                    landmarks=[],
                )
            )

        img = CanonicalImage(
            image_id="img_1",
            file_path=file_path,
            width=width,
            height=height,
            objects=objects,
        )
        return CanonicalDataset(images=[img])

    @classmethod
    def parse_file(
        cls,
        file_path: str,
        image_path: Optional[str] = None,
        width: int = 0,
        height: int = 0,
        class_names: Optional[Dict[int, str]] = None,
    ) -> CanonicalDataset:
        if image_path is None:
            base, _ = os.path.splitext(file_path)
            image_path = f"{base}.jpg"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.parse_string(
            content,
            file_path=image_path,
            width=width,
            height=height,
            class_names=class_names,
        )

    @classmethod
    def parse_directory(
        cls,
        txt_dir: str,
        image_dir: Optional[str] = None,
        class_names: Optional[Dict[int, str]] = None,
    ) -> CanonicalDataset:
        images = []
        if not os.path.exists(txt_dir):
            return CanonicalDataset(images=[])

        txt_files = sorted([f for f in os.listdir(txt_dir) if f.endswith(".txt")])
        for idx, txt_file in enumerate(txt_files):
            txt_path = os.path.join(txt_dir, txt_file)
            base_name = os.path.splitext(txt_file)[0]
            img_rel_path = f"{base_name}.jpg"
            if image_dir:
                img_rel_path = os.path.join(image_dir, img_rel_path)

            ds = cls.parse_file(
                txt_path,
                image_path=img_rel_path,
                class_names=class_names,
            )
            if ds.images:
                img = ds.images[0]
                img.image_id = f"img_{idx + 1}"
                images.append(img)

        return CanonicalDataset(images=images)
