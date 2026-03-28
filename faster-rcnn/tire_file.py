#pip install torch torchvision opencv-python deep-sort-realtime numpy
import os
import cv2
import glob
import torch
import torchvision
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.ops import nms
import random

# --- Configuration ---
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(DEVICE)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "tire_video.mp4")
DATA_DIR = os.path.join(BASE_DIR, "data")
CLASSES = {"tire": 1}

def setup_directories():
    for folder in ['video', 'frames', 'labels']:
        os.makedirs(os.path.join(DATA_DIR, folder), exist_ok=True)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{DATA_DIR}/frames/frame_{count:06d}.jpg", frame)
        count += 1
    cap.release()
    return count

class TireDataset(Dataset):
    def __init__(self, frame_dir, label_dir, transforms=None, augment=True):
        img_files = {os.path.splitext(f)[0]: os.path.join(frame_dir, f) 
                     for f in os.listdir(frame_dir) if f.endswith('.jpg')}
        xml_files = {os.path.splitext(f)[0]: os.path.join(label_dir, f) 
                     for f in os.listdir(label_dir) if f.endswith('.xml')}

        common_names = sorted(list(set(img_files.keys()) & set(xml_files.keys())))
        
        # Filter out XMLs that have no bounding boxes (empty annotations from CVAT)
        self.frame_paths = []
        self.label_paths = []
        skipped = 0
        for name in common_names:
            tree = ET.parse(xml_files[name])
            root = tree.getroot()
            has_objects = any(
                obj.find('name').text in CLASSES for obj in root.iter('object')
            )
            if has_objects:
                self.frame_paths.append(img_files[name])
                self.label_paths.append(xml_files[name])
            else:
                skipped += 1
        
        if skipped > 0:
            print(f"Skipped {skipped} frames with no valid annotations.")
        
        self.transforms = transforms or T.ToTensor()
        self.augment = augment

    def _augment(self, img, boxes):
        """Apply data augmentation to increase effective dataset size."""
        w, h = img.size
        
        # Random horizontal flip (50% chance)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes) > 0:
                boxes_arr = np.array(boxes)
                boxes_arr[:, [0, 2]] = w - boxes_arr[:, [2, 0]]
                boxes = boxes_arr.tolist()
        
        # Random brightness/contrast adjustment
        if random.random() > 0.5:
            from torchvision.transforms import functional as F
            factor = random.uniform(0.7, 1.3)
            img = F.adjust_brightness(img, factor)
        
        if random.random() > 0.5:
            from torchvision.transforms import functional as F
            factor = random.uniform(0.7, 1.3)
            img = F.adjust_contrast(img, factor)
        
        # Random slight color jitter
        if random.random() > 0.5:
            from torchvision.transforms import functional as F
            factor = random.uniform(0.8, 1.2)
            img = F.adjust_saturation(img, factor)
        
        return img, boxes

    def __getitem__(self, idx):
        img = Image.open(self.frame_paths[idx]).convert("RGB")
        tree = ET.parse(self.label_paths[idx])
        root = tree.getroot()
        
        boxes, labels = [], []
        for obj in root.iter('object'):
            label = obj.find('name').text
            if label in CLASSES:
                xmlbox = obj.find('bndbox')
                b = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                     float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)]
                # Validate box dimensions
                if b[2] > b[0] and b[3] > b[1]:
                    boxes.append(b)
                    labels.append(CLASSES[label])
        
        # Apply augmentation during training
        if self.augment and len(boxes) > 0:
            img, boxes = self._augment(img, boxes)
        
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }
        return self.transforms(img), target

    def __len__(self):
        return len(self.frame_paths)

def get_model(num_classes):
    # Load a pre-trained backbone (ResNet50-FPN is much better for detection)
    backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"  # Pre-trained on COCO
    )
    
    # Replace the classifier head for our number of classes
    in_features = backbone.roi_heads.box_predictor.cls_score.in_features
    backbone.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    # Tune RPN parameters for small objects like tires
    # Lower NMS threshold in RPN to reduce duplicate proposals
    backbone.rpn.nms_thresh = 0.5  # was 0.7 by default — tighter filtering
    
    # Increase number of proposals to catch small objects
    backbone.rpn._pre_nms_top_n['training'] = 2000
    backbone.rpn._post_nms_top_n['training'] = 1000
    backbone.rpn._pre_nms_top_n['testing'] = 1000
    backbone.rpn._post_nms_top_n['testing'] = 500
    
    # Tighten the box detector NMS threshold
    backbone.roi_heads.nms_thresh = 0.3  # default 0.5 — reduces duplicate boxes
    
    # Higher score threshold to reject low-confidence predictions
    backbone.roi_heads.score_thresh = 0.5  # default 0.05
    
    return backbone

def run_inference(model, video_path, output_path="output.mp4"):
    tracker = DeepSort(
        max_age=30,       # tracks can survive 30 frames without detection
        n_init=3,          # require 3 consecutive detections to confirm a track
        max_iou_distance=0.7,
        nn_budget=30
    )
    cap = cv2.VideoCapture(video_path)
    
    # Get source video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path.replace('.mp4', '.avi'), fourcc, fps, (width, height))
    
    transform = T.ToTensor()
    conf_threshold = 0.75       # Higher confidence for cleaner detections
    nms_iou_threshold = 0.25    # Tighter NMS to remove overlapping boxes
    min_box_size = 15           # minimum width/height in pixels
    max_box_size = 300          # maximum width/height (tires shouldn't be huge)
    max_box_area_ratio = 0.15   # box area shouldn't exceed 15% of frame
    frame_count = 0
    frame_area = width * height

    model.eval()
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count}/{total_frames}...")
            
            # Convert BGR frame to RGB tensor
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img_rgb).to(DEVICE)
            
            # Run detection
            predictions = model([img_tensor])
            pred = predictions[0]
            
            boxes = pred['boxes']
            scores = pred['scores']
            labels = pred['labels']
            
            # Step 1: Filter by confidence threshold
            mask = scores >= conf_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            # Step 2: Apply NMS to remove overlapping boxes
            if len(boxes) > 0:
                keep = nms(boxes, scores, nms_iou_threshold)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
            
            boxes_np = boxes.cpu().numpy()
            scores_np = scores.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Step 3: Filter out unreasonable boxes (too tiny, too large, wrong aspect ratio)
            valid = []
            for i, box in enumerate(boxes_np):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                box_area = w * h
                aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                
                # Size constraints
                if w < min_box_size or h < min_box_size:
                    continue
                if w > max_box_size or h > max_box_size:
                    continue
                # Area ratio constraint — tire won't occupy huge portion of frame
                if box_area / frame_area > max_box_area_ratio:
                    continue
                # Aspect ratio constraint — tire is roughly circular/square, not super elongated
                if aspect_ratio > 4.0:
                    continue
                
                valid.append(i)
            
            boxes_np = boxes_np[valid]
            scores_np = scores_np[valid]
            labels_np = labels_np[valid]
            
            # Debug: print detection info for first 20 frames
            if frame_count <= 20:
                print(f"  Frame {frame_count}: {len(scores_np)} detections "
                      f"(conf>={conf_threshold}, NMS_IoU={nms_iou_threshold})")
                for i, (box, score) in enumerate(zip(boxes_np, scores_np)):
                    x1, y1, x2, y2 = box
                    print(f"    Det {i}: [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] "
                          f"conf={score:.3f} size={x2-x1:.0f}x{y2-y1:.0f}")
            
            # Format detections for DeepSort: list of ([x1, y1, w, h], confidence, class)
            detections = []
            for box, score, label in zip(boxes_np, scores_np, labels_np):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], score, 'tire'))

            # Update tracker
            tracks = tracker.update_tracks(detections, frame=frame)
            
            # Draw tracked objects on frame
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()  # [left, top, right, bottom]
                x1, y1, x2, y2 = [int(v) for v in ltrb]
                
                # Additional check: don't draw unreasonably large tracked boxes
                tw, th = x2 - x1, y2 - y1
                if tw > max_box_size or th > max_box_size:
                    continue
                if tw * th / frame_area > max_box_area_ratio:
                    continue
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw label with track ID
                label_text = f"Tire ID:{track_id}"
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(frame)
    
    cap.release()
    out.release()
    print(f"Output saved to: {output_path.replace('.mp4', '.avi')}")

def train_model(model, dataloader, num_epochs=30):
    """Training loop with learning rate scheduler for better convergence."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler: reduce LR when loss plateaus
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in dataloader:
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            epoch_loss += losses.item()
        
        lr_scheduler.step()
        
        if len(dataloader) > 0:
            avg_loss = epoch_loss / len(dataloader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")
    
    return model

if __name__ == "__main__":
    setup_directories()
    
    frames_dir = os.path.join(DATA_DIR, "frames")
    existing_frames = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')] if os.path.exists(frames_dir) else []
    
    if len(existing_frames) > 0:
        print(f"Found {len(existing_frames)} existing frames. Skipping extraction.")
    elif os.path.exists(VIDEO_PATH):
        print(f"Extracting frames from {VIDEO_PATH}...")
        extract_frames(VIDEO_PATH)
    else:
        print(f"Video '{VIDEO_PATH}' not found. Skipping extraction.")
    
    model = get_model(num_classes=2) # 1 class + background
    model.to(DEVICE)
    print("Model initialized.")

    model_save_path = os.path.join(BASE_DIR, "trained_model.pth")
    
    # If a trained model exists, load it and skip training
    if os.path.exists(model_save_path):
        print(f"Loading saved model from {model_save_path}...")
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        print("Model loaded successfully. Skipping training.")
    else:
        dataset = TireDataset(os.path.join(DATA_DIR, "frames"), os.path.join(DATA_DIR, "labels"), augment=True)
        
        if len(dataset) > 0:
            print(f"Found {len(dataset)} labeled frames. Starting training...")
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))
            model = train_model(model, dataloader, num_epochs=30)
            # Save trained weights so you don't have to retrain every time
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
            print("Training complete.")
        else:
            print("No labeled data found for training. Please add XML labels to 'data/labels' to train.")

    if os.path.exists(VIDEO_PATH):
        output_file = "output.avi"
        print(f"Running inference to process the video... Saving to {output_file}")
        run_inference(model, VIDEO_PATH, output_file)
        print("Processing complete.")
    else:
        print("No video found for processing.")