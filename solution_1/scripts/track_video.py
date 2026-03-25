import os
from ultralytics import YOLO

def run_tracking():
    # 1. Load your exported ONNX or PyTorch model
    # Usually you want the best.pt or best.onnx from your training run!
    model_path = "../web/models/best.onnx"
    video_path = "../web/tire_video.mp4"
    output_path = "../web/tire_video_tracked.mp4"

    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}. Make sure to download best.onnx from Colab first!")
        return
        
    if not os.path.exists(video_path):
        print(f"Error: Could not find video at {video_path}")
        return

    print("Loading YOLO model...")
    model = YOLO(model_path, task='detect')

    print("Starting multi-object tracking via ByteTrack...")
    
    # 2. Layering a Multi-Object Tracker (MOT) over your detector!
    # By running model.track(), Ultralytics natively spawns a Kalman Filter for your boxes.
    # imgsz=1024 allows the detector to see fine details.
    # tracker="bytetrack.yaml" links boxes between frames even when the detector drops them for a few frames.
    # tracker="botsort.yaml" is another great alternative!
    
    results = model.track(
        source=video_path,
        imgsz=1024,
        tracker="bytetrack.yaml", # The magic physics/velocity tracker
        conf=0.3,                 # Base confidence 
        iou=0.5,                  # NMS overlap setting (not strictly necessary for YOLOv10, but standard)
        save=True,                # Save the resulting video!
        name="tracked_output"     # Subfolder name for output
    )
    
    print("Tracking complete! Check the 'runs/detect/tracked_output' folder for your tracked video.")

if __name__ == "__main__":
    run_tracking()
