import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import ArteryDataset
from custom_model import CustomArteryDetector
import cv2
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

def visualize_prediction(image, gt_box, pred_box, confidence, save_path):
    """Visualize prediction and ground truth on the image"""
    # Convert image to color (assuming grayscale input)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = image.copy()
    
    height, width = image.shape[:2]
    
    # Draw ground truth box (green)
    if gt_box is not None:
        x, y, w, h = gt_box
        x1 = int(x * width - w * width / 2)
        y1 = int(y * height - h * height / 2) 
        x2 = int(x * width + w * width / 2)
        y2 = int(y * height + h * height / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, 'GT', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw prediction box (blue)
    if pred_box is not None:
        x, y, w, h = pred_box
        x1 = int(x * width - w * width / 2)
        y1 = int(y * height - h * height / 2)
        x2 = int(x * width + w * width / 2)
        y2 = int(y * height + h * height / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f'Pred: {confidence:.2f}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Save the image
    cv2.imwrite(save_path, image)

def compute_iou(boxA, boxB, image_size=(1.0, 1.0)):
    """
    Compute IoU between two boxes in normalized coordinates
    boxA, boxB: [x_center, y_center, width, height]
    """
    width, height = image_size
    
    # Convert to corner coordinates
    boxA_x1 = boxA[0] - boxA[2]/2
    boxA_y1 = boxA[1] - boxA[3]/2
    boxA_x2 = boxA[0] + boxA[2]/2
    boxA_y2 = boxA[1] + boxA[3]/2
    
    boxB_x1 = boxB[0] - boxB[2]/2
    boxB_y1 = boxB[1] - boxB[3]/2
    boxB_x2 = boxB[0] + boxB[2]/2
    boxB_y2 = boxB[1] + boxB[3]/2
    
    # Determine the coordinates of the intersection rectangle
    x_left = max(boxA_x1, boxB_x1)
    y_top = max(boxA_y1, boxB_y1)
    x_right = min(boxA_x2, boxB_x2)
    y_bottom = min(boxA_y2, boxB_y2)
    
    # Check if there's an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Compute intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute union area
    boxA_area = (boxA_x2 - boxA_x1) * (boxA_y2 - boxA_y1)
    boxB_area = (boxB_x2 - boxB_x1) * (boxB_y2 - boxB_y1)
    union_area = boxA_area + boxB_area - intersection_area
    
    # Compute IoU
    iou = intersection_area / union_area
    
    return max(0, iou)

def evaluate_model(model_path, data_path, annotation_type='auto', batch_size=8, conf_threshold=0.5, iou_threshold=0.5):
    """Evaluate the model on the test dataset and compute metrics"""
    # Create output directory for visualizations
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}")
    if model_path.endswith('.pth'):
        # Check if it's a full model or just state dict
        try:
            model = torch.load(model_path, map_location=device)
            if not isinstance(model, nn.Module):
                # It's a state dict or checkpoint
                checkpoint = torch.load(model_path, map_location=device)
                model = CustomArteryDetector().to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
        except:
            # Try loading as state dict directly
            model = CustomArteryDetector().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise ValueError(f"Unsupported model format: {model_path}")
    
    model.eval()
    
    # Load test dataset
    test_dataset = ArteryDataset(data_path, split='test', annotation_type=annotation_type, filter_empty=False)
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluation metrics
    y_true = []  # Ground truth labels (0/1)
    y_scores = []  # Predicted confidence scores
    
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    tn = 0  # True negatives
    
    # IoU values for correct detections
    ious = []
    
    # Sample images for visualization
    vis_count = 0
    max_vis = 20  # Maximum visualizations to save
    
    # Process each batch
    with torch.no_grad():
        for batch_idx, (images, target_locs, target_conf) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            target_locs = target_locs.to(device)
            target_conf = target_conf.to(device)
            
            # Get predictions
            pred_locs, pred_conf = model(images)
            
            # Process each image in batch
            for i in range(images.size(0)):
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                
                # Get ground truth
                gt_has_artery = target_conf[i].item() > 0.5
                gt_box = target_locs[i].cpu().numpy() if gt_has_artery else None
                
                # Get prediction
                pred_has_artery = pred_conf[i].item() > conf_threshold
                pred_box = pred_locs[i].cpu().numpy() if pred_has_artery else None
                
                # Store for precision-recall curve
                y_true.append(1 if gt_has_artery else 0)
                y_scores.append(pred_conf[i].item())
                
                # Calculate IoU if both have detections
                iou = 0
                if gt_has_artery and pred_has_artery:
                    iou = compute_iou(gt_box, pred_box)
                    ious.append(iou)
                
                # Update confusion matrix
                if gt_has_artery and pred_has_artery:
                    if iou >= iou_threshold:
                        tp += 1
                    else:
                        fp += 1  # Predicted artery but wrong location
                        fn += 1  # Missed the correct artery
                elif gt_has_artery and not pred_has_artery:
                    fn += 1
                elif not gt_has_artery and pred_has_artery:
                    fp += 1
                else:  # not gt_has_artery and not pred_has_artery
                    tn += 1
                
                # Visualize some predictions
                if vis_count < max_vis:
                    # Choose strategy for visualization:
                    # 1. Always save first N images
                    # 2. Save on interesting cases (TP, FP, FN)
                    # 3. Save randomly
                    
                    # Strategy: Save instances of TP, FP, FN, and some TN
                    save_vis = False
                    
                    if gt_has_artery and pred_has_artery and iou >= iou_threshold:  # TP
                        save_vis = True
                        result_type = "TP"
                    elif gt_has_artery and not pred_has_artery:  # FN
                        save_vis = True
                        result_type = "FN"
                    elif not gt_has_artery and pred_has_artery:  # FP
                        save_vis = True
                        result_type = "FP"
                    elif not gt_has_artery and not pred_has_artery and vis_count < 5:  # Some TN
                        save_vis = True
                        result_type = "TN"
                    
                    if save_vis:
                        # Prepare the image for visualization (handle grayscale/color)
                        if img.shape[2] == 1:  # Grayscale
                            vis_img = img[:, :, 0]
                        else:  # Color
                            vis_img = img
                            
                        # Save visualization
                        conf_score = pred_conf[i].item() if pred_has_artery else 0
                        vis_path = f'evaluation_results/{result_type}_{batch_idx}_{i}_conf{conf_score:.2f}.jpg'
                        visualize_prediction(vis_img, gt_box, pred_box, conf_score, vis_path)
                        vis_count += 1
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Calculate mean IoU for correct detections
    mean_iou = np.mean(ious) if ious else 0
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Confidence threshold: {conf_threshold}, IoU threshold: {iou_threshold}")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    
    # Calculate and plot precision-recall curve
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall_curve, precision_curve, label=f'AP={average_precision:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('evaluation_results/precision_recall_curve.png')
    
    # Calculate and save confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig('evaluation_results/confusion_matrix.png')
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'average_precision': average_precision
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate artery detection model")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_custom_model.pth", help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="full artery data 85 10 5 images txt xml json", help="Path to dataset")
    parser.add_argument("--annotation_type", type=str, default="auto", choices=["xml", "txt", "auto"], help="Annotation type")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    args = parser.parse_args()
    
    evaluate_model(
        args.model_path,
        args.data_path,
        args.annotation_type,
        args.batch_size,
        args.conf_threshold,
        args.iou_threshold
    ) 
