import subprocess
import os

def extract_frames(video_path, output_dir, fps=2):
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    
    # Utilizing FFmpeg for hardware-accelerated, high-efficiency extraction
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",  # High quality JPEG encoding
        output_pattern
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Frames extracted successfully to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg extraction failed: {e.stderr.decode()}")

if __name__ == "__main__":
    # Build absolute paths relative to the project root (one level up from scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")
    
    video_path = os.path.join(project_root, "data", "raw_video", "tire_video.mp4")
    output_dir = os.path.join(project_root, "data", "frames")
    
    extract_frames(video_path, output_dir)