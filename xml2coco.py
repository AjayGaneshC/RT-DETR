import os
import json
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from PIL import Image




# Paths - Change these to match your dataset
IMAGE_DIR = "/home/htic/Wade-Archives/PolypsSet/train2019/Image/"
XML_DIR = "/home/htic/Wade-Archives/PolypsSet/train2019/Annotation/"
OUTPUT_JSON = "/home/htic/Wade-Archives/PolypsSet/train2019/train_annotations.json"

# Ensure image and annotation files match
image_filenames = {f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")}
xml_filenames = {f for f in os.listdir(XML_DIR) if f.endswith(".xml")}

# COCO format structure
coco = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "polyp"}],  # Modify class name if needed
}

# Annotation counters
image_id = 0
annotation_id = 0

for xml_file in tqdm(sorted(xml_filenames), desc="Converting"):
    xml_path = os.path.join(XML_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image filename from XML
    filename = root.find("filename").text.strip()

    # If image filename in XML is different, rename it to match numerical order
    xml_base_name = os.path.splitext(xml_file)[0]
    expected_filename = f"{xml_base_name}.jpg"

    if filename not in image_filenames:
        print(f"Warning: No matching image for {filename} -> Expected: {expected_filename}")
        filename = expected_filename  # Fix the mismatch

    if filename not in image_filenames:
        continue  # Skip if still no matching image

    # Get image width & height
    size_tag = root.find("size")
    width = int(size_tag.find("width").text)
    height = int(size_tag.find("height").text)

    # Add image entry
    coco["images"].append({
        "id": image_id,
        "file_name": filename,
        "width": width,
        "height": height
    })

    # Process bounding boxes
    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()

        # Modify if you have multiple classes
        category_id = 1  # Since you have only one class (polyp)

        # Get bounding box
        bbox_tag = obj.find("bndbox")
        xmin = int(float(bbox_tag.find("xmin").text))
        ymin = int(float(bbox_tag.find("ymin").text))
        xmax = int(float(bbox_tag.find("xmax").text))
        ymax = int(float(bbox_tag.find("ymax").text))
        w = xmax - xmin
        h = ymax - ymin

        # Add annotation
        coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [xmin, ymin, w, h],
            "area": w * h,
            "iscrowd": 0
        })
        annotation_id += 1

    image_id += 1

# Save JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(coco, f, indent=4)

print(f"COCO annotations saved to {OUTPUT_JSON}")
