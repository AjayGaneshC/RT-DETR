import os
import json
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm
import re

def find_xml_files(root_dir):
    """ Recursively find all XML annotation files in subdirectories. """
    xml_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".xml"):
                xml_files.append(os.path.join(dirpath, file))
    return xml_files

def get_label2id(xml_files):
    """ Extract unique labels from XML annotations and assign numeric IDs. """
    labels = set()
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall("object"):
            labels.add(obj.findtext("name"))
    
    label2id = {label: idx + 1 for idx, label in enumerate(sorted(labels))}
    print(f"📌 Detected Labels: {label2id}")
    return label2id

def get_image_info(annotation_root, extract_num_from_imgid=True):
    """ Extract image metadata from XML. """
    filename = annotation_root.findtext("filename")
    if filename is None:
        filename = annotation_root.findtext("path")
    filename = os.path.basename(filename)

    img_id = os.path.splitext(filename)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r"\d+", img_id)[0])

    size = annotation_root.find("size")
    width, height = int(size.findtext("width")), int(size.findtext("height"))

    return {"file_name": filename, "height": height, "width": width, "id": img_id}

def get_coco_annotation_from_obj(obj, label2id, ann_id):
    """ Convert XML annotation to COCO format. """
    label = obj.findtext("name")
    category_id = label2id[label]
    bndbox = obj.find("bndbox")
    xmin, ymin = int(bndbox.findtext("xmin")) - 1, int(bndbox.findtext("ymin")) - 1
    xmax, ymax = int(bndbox.findtext("xmax")), int(bndbox.findtext("ymax"))
    
    assert xmax > xmin and ymax > ymin, f"Invalid box size: {xmin, ymin, xmax, ymax}"

    return {
        "id": ann_id,
        "image_id": None,  # Will be set later
        "category_id": category_id,
        "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
        "area": (xmax - xmin) * (ymax - ymin),
        "iscrowd": 0,
        "segmentation": []
    }

def convert_xmls_to_cocojson(xml_files, output_jsonpath):
    """ Convert XML annotations to COCO JSON format. """
    label2id = get_label2id(xml_files)
    coco_json = {"images": [], "annotations": [], "categories": []}

    ann_id = 1
    for xml_file in tqdm(xml_files, desc="Converting"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        img_info = get_image_info(root)
        coco_json["images"].append(img_info)

        for obj in root.findall("object"):
            annotation = get_coco_annotation_from_obj(obj, label2id, ann_id)
            annotation["image_id"] = img_info["id"]
            coco_json["annotations"].append(annotation)
            ann_id += 1

    for label, label_id in label2id.items():
        coco_json["categories"].append({"supercategory": "none", "id": label_id, "name": label})

    with open(output_jsonpath, "w") as f:
        json.dump(coco_json, f, indent=4)
    
    print(f"✅ COCO annotations saved to {output_jsonpath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VOC XML annotations to COCO JSON format")
    parser.add_argument("--ann_dir", type=str, required=True, help="Path to annotation directory (e.g., test/ or val/)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    xml_files = find_xml_files(args.ann_dir)
    if not xml_files:
        print("❌ No XML files found in the specified directory!")
        exit(1)

    convert_xmls_to_cocojson(xml_files, args.output)
