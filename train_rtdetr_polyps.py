import os
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor, TrainingArguments, Trainer
from datasets import load_dataset
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import json
import random
import sys
from transformers import DefaultDataCollator

class PolypDataset(Dataset):
    def __init__(self, image_dir, annotation_file, image_processor, train=True):
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.train = train
        self.cache = {}
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        print(f"Total annotations: {len(self.annotations['annotations'])}")
        print(f"Total images in annotations: {len(self.annotations['images'])}")
            
        # Create image id to annotations mapping
        self.image_to_annotations = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = []
            self.image_to_annotations[img_id].append(ann)
        
        print(f"Number of images with annotations: {len(self.image_to_annotations)}")
        
        # Filter out images that don't exist
        self.valid_image_ids = []
        skipped_images = 0
        missing_paths = []
        
        for img_id in self.image_to_annotations.keys():
            img_info = next(img for img in self.annotations['images'] if img['id'] == img_id)
            
            # Comprehensive path resolution strategy
            possible_base_dirs = [
                os.path.join(self.image_dir, 'Image'),  # Primary image directory
            ]
            
            # For validation, add numbered subdirectories
            if not self.train:
                possible_base_dirs.extend([
                    os.path.join(self.image_dir, 'Image', str(subdir))
                    for subdir in range(1, 18)  # Subdirectories 1-17
                ])
            
            # Possible paths to try
            paths_to_try = [
                os.path.join(base_dir, img_info['file_name'])
                for base_dir in possible_base_dirs
            ]
            
            # Additional variations
            paths_to_try.extend([
                os.path.join(base_dir, f'Image_{img_info["file_name"]}')
                for base_dir in possible_base_dirs
            ])
            
            image_exists = False
            valid_path = None
            for path in paths_to_try:
                if os.path.exists(path):
                    image_exists = True
                    valid_path = path
                    break
                    
            if image_exists:
                self.valid_image_ids.append((img_id, valid_path))
            else:
                skipped_images += 1
                missing_paths.append(paths_to_try)
        
        print(f"{'Train' if train else 'Val'} dataset size: {len(self.valid_image_ids)}")
        if skipped_images > 0:
            print(f"Skipped {skipped_images} images due to missing files")
            print("First 5 missing image paths:")
            for paths in missing_paths[:5]:
                print(f"  Tried: {paths}")
        
        # Raise an error if no images are found
        if len(self.valid_image_ids) == 0:
            raise ValueError(f"No images found in {self.image_dir}. Check your dataset paths.")
    
    def __len__(self):
        return len(self.valid_image_ids)
    
    def __getitem__(self, idx):
        img_id, image_path = self.valid_image_ids[idx]
        
        # Return cached result if available
        if img_id in self.cache:
            return self.cache[img_id]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            raise
        
        # Get annotations for this image
        anns = self.image_to_annotations[img_id]
        
        # Prepare boxes and labels
        boxes = []
        labels = []
        areas = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # Convert to x1,y1,x2,y2 format
            labels.append(ann['category_id'])
            areas.append(w * h)  # Calculate area as width * height
            
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        areas = np.array(areas, dtype=np.float32)
        
        # Prepare inputs
        encoding = self.image_processor(
            images=image,
            annotations={
                "image_id": img_id,
                "annotations": [
                    {
                        "bbox": box,
                        "category_id": label,
                        "bbox_mode": "xyxy",
                        "area": area,
                        "iscrowd": 0
                    } for box, label, area in zip(boxes, labels, areas)
                ]
            },
            return_tensors="pt"
        )
        
        # Cache and return the result
        result = {
            "pixel_values": encoding.pixel_values[0],
            "labels": encoding.labels[0]
        }
        self.cache[img_id] = result
        return result

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = [example["labels"] for example in batch]
    
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }

def main():
    # Print system information
    print("\n=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version (torch): {torch.version.cuda}")
    
    print("\n=== CUDA Information ===")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load model and processor
    print("\n=== Loading Model ===")
    model = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd")
    image_processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_v2_r50vd")
    
    model = model.to(device)
    
    print("\nCreating datasets...")
    # Create datasets
    train_dataset = PolypDataset(
        image_dir="/home/htic/Wade-Archives/PolypsSet/train2019",
        annotation_file="/home/htic/Wade-Archives/PolypsSet/train2019/train2019_annotation.json",
        image_processor=image_processor,
        train=True
    )
    
    val_dataset = PolypDataset(
        image_dir="/home/htic/Wade-Archives/PolypsSet/val2019",
        annotation_file="/home/htic/Wade-Archives/PolypsSet/val2019/val2019_annotation.json",
        image_processor=image_processor,
        train=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="rtdetr_polyps_results",
        num_train_epochs=50,
        learning_rate=2e-4,
        weight_decay=1e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        remove_unused_columns=False,
        label_names=["labels", "boxes"],
        logging_steps=10,
        gradient_accumulation_steps=4,
        fp16=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    print("\nStarting training...")
    trainer.train()

if __name__ == "__main__":
    main()
