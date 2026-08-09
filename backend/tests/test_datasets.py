import os
import tempfile
import pytest
from backend.datasets.canonical import (
    LandmarkPoint,
    CanonicalObject,
    CanonicalImage,
    CanonicalDataset,
)
from backend.datasets.importers import (
    TPSImporter,
    DlibXMLImporter,
    YOLOOBBImporter,
)


def test_landmark_point_serialization():
    pt = LandmarkPoint(name="snout", x=12.5, y=34.0)
    d = pt.to_dict()
    assert d == {"name": "snout", "x": 12.5, "y": 34.0}
    restored = LandmarkPoint.from_dict(d)
    assert restored == pt


def test_canonical_object_serialization():
    pt1 = LandmarkPoint(name="0", x=10.0, y=10.0)
    pt2 = LandmarkPoint(name="1", x=50.0, y=30.0)
    obj = CanonicalObject(
        object_id="obj_1",
        class_name="lizard",
        obb=[30.0, 20.0, 48.0, 24.0, 0.0],
        landmarks=[pt1, pt2],
        specimen_id="spec_001",
    )
    d = obj.to_dict()
    assert d["object_id"] == "obj_1"
    assert d["class_name"] == "lizard"
    assert len(d["landmarks"]) == 2
    assert d["specimen_id"] == "spec_001"
    restored = CanonicalObject.from_dict(d)
    assert restored == obj


def test_canonical_image_and_dataset_serialization():
    pt = LandmarkPoint(name="0", x=10.0, y=10.0)
    obj = CanonicalObject(
        object_id="obj_1",
        class_name="lizard",
        obb=[10.0, 10.0, 20.0, 20.0, 0.0],
        landmarks=[pt],
        specimen_id="spec_1",
    )
    img = CanonicalImage(
        image_id="img_1",
        file_path="sample.jpg",
        width=800,
        height=600,
        objects=[obj],
    )
    ds = CanonicalDataset(images=[img])

    d = ds.to_dict()
    restored = CanonicalDataset.from_dict(d)
    assert len(restored.images) == 1
    assert restored.images[0].file_path == "sample.jpg"
    assert len(restored.images[0].objects) == 1
    assert restored.images[0].objects[0].landmarks[0].name == "0"


def test_derive_obb_from_landmarks():
    pts = [(10.0, 10.0), (50.0, 30.0)]
    obb = CanonicalDataset.derive_obb_from_landmarks(pts, padding=0.2)
    # min_x=10, max_x=50 (w=40), min_y=10, max_y=30 (h=20)
    # cx=30, cy=20, w_padded=40*1.2=48, h_padded=20*1.2=24, angle=0.0
    assert obb == [30.0, 20.0, 48.0, 24.0, 0.0]

    empty_obb = CanonicalDataset.derive_obb_from_landmarks([], padding=0.2)
    assert empty_obb == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_tps_importer_string():
    tps_content = """LM=3
10.0 20.0
30.0 40.0
50.0 60.0
IMAGE=lizard_01.jpg
ID=spec_123
SCALE=0.01
"""
    ds = TPSImporter.parse_string(tps_content)
    assert len(ds.images) == 1
    img = ds.images[0]
    assert img.file_path == "lizard_01.jpg"
    assert len(img.objects) == 1
    obj = img.objects[0]
    assert obj.specimen_id == "spec_123"
    assert len(obj.landmarks) == 3
    assert obj.landmarks[0].x == 10.0
    assert obj.landmarks[0].y == 20.0
    assert obj.landmarks[2].x == 50.0
    assert obj.landmarks[2].y == 60.0
    # Check OBB derivation: min_x=10, max_x=50 (w=40), min_y=20, max_y=60 (h=40)
    # cx=30, cy=40, padded w=48, padded h=48
    assert obj.obb == [30.0, 40.0, 48.0, 48.0, 0.0]


def test_tps_importer_file(tmp_path):
    tps_file = tmp_path / "test.tps"
    tps_file.write_text("""LM=2
1.0 2.0
3.0 4.0
IMAGE=imgA.jpg
ID=specA

LM=2
5.0 6.0
7.0 8.0
IMAGE=imgB.jpg
ID=specB
""")
    ds = TPSImporter.parse_file(str(tps_file))
    assert len(ds.images) == 2
    assert ds.images[0].file_path == "imgA.jpg"
    assert ds.images[1].file_path == "imgB.jpg"


def test_dlib_xml_importer_string():
    xml_content = """<?xml version='1.0' encoding='UTF-8'?>
<dataset>
  <name>Test Dlib</name>
  <images>
    <image file="specimen1.jpg" width="1000" height="800">
      <box top="50" left="100" width="200" height="150" label="head">
        <part name="0" x="120" y="80"/>
        <part name="1" x="150" y="90"/>
      </box>
    </image>
  </images>
</dataset>
"""
    ds = DlibXMLImporter.parse_string(xml_content)
    assert len(ds.images) == 1
    img = ds.images[0]
    assert img.file_path == "specimen1.jpg"
    assert img.width == 1000
    assert img.height == 800
    assert len(img.objects) == 1
    obj = img.objects[0]
    assert obj.class_name == "head"
    # left=100, top=50, w=200, h=150 -> cx=200, cy=125
    assert obj.obb == [200.0, 125.0, 200.0, 150.0, 0.0]
    assert len(obj.landmarks) == 2
    assert obj.landmarks[0].name == "0"
    assert obj.landmarks[0].x == 120.0
    assert obj.landmarks[0].y == 80.0


def test_dlib_xml_importer_file(tmp_path):
    xml_file = tmp_path / "annotations.xml"
    xml_file.write_text("""<?xml version='1.0' encoding='UTF-8'?>
<dataset>
  <images>
    <image file="file1.jpg">
      <box top="10" left="10" width="50" height="50">
        <part name="0" x="15" y="15"/>
      </box>
    </image>
  </images>
</dataset>
""")
    ds = DlibXMLImporter.parse_file(str(xml_file))
    assert len(ds.images) == 1
    assert ds.images[0].file_path == "file1.jpg"


def test_yolo_obb_importer_string():
    # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
    yolo_content = "0 10.0 10.0 50.0 10.0 50.0 40.0 10.0 40.0\n1 100.0 100.0 200.0 100.0 200.0 150.0 100.0 150.0"
    class_names = {0: "lizard", 1: "scale"}
    ds = YOLOOBBImporter.parse_string(
        yolo_content,
        file_path="image1.jpg",
        width=800,
        height=600,
        class_names=class_names,
    )
    assert len(ds.images) == 1
    img = ds.images[0]
    assert img.file_path == "image1.jpg"
    assert len(img.objects) == 2
    obj0 = img.objects[0]
    assert obj0.class_name == "lizard"
    # Corners (10,10), (50,10), (50,40), (10,40)
    # cx=30, cy=25, w=40, h=30, angle=0.0
    assert obj0.obb == [30.0, 25.0, 40.0, 30.0, 0.0]

    obj1 = img.objects[1]
    assert obj1.class_name == "scale"
    # Corners (100,100), (200,100), (200,150), (100,150)
    # cx=150, cy=125, w=100, h=50, angle=0.0
    assert obj1.obb == [150.0, 125.0, 100.0, 50.0, 0.0]


def test_yolo_obb_importer_normalized_coords():
    # Normalized coordinates 0..1 with width=100, height=100
    yolo_content = "0 0.1 0.1 0.5 0.1 0.5 0.4 0.1 0.4"
    ds = YOLOOBBImporter.parse_string(
        yolo_content,
        file_path="image_norm.jpg",
        width=100,
        height=100,
        class_names={0: "lizard"},
    )
    obj = ds.images[0].objects[0]
    assert obj.obb == [30.0, 25.0, 40.0, 30.0, 0.0]


def test_yolo_obb_importer_file(tmp_path):
    txt_file = tmp_path / "img1.txt"
    txt_file.write_text("0 10.0 10.0 50.0 10.0 50.0 40.0 10.0 40.0\n")
    ds = YOLOOBBImporter.parse_file(
        str(txt_file),
        image_path="img1.jpg",
        width=800,
        height=600,
        class_names={0: "lizard"},
    )
    assert len(ds.images) == 1
    assert ds.images[0].file_path == "img1.jpg"
