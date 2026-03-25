import torch
import torch.nn as nn
import sys
import os

# Point to the cloned OSTrack repo using robust absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
ostrack_path = os.path.join(parent_dir, 'OSTrack')

sys.path.insert(0, ostrack_path)

from lib.models.ostrack import build_ostrack
from lib.config.ostrack.config import cfg, update_config_from_file


def load_model(config_path, checkpoint_path):
    """Load OSTrack model with proper config and weights."""
    
    # Load config
    update_config_from_file(config_path)
    
    # Build model from config
    model = build_ostrack(cfg, training=False)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'net' in checkpoint:
        state_dict = checkpoint['net']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded successfully!")
    return model


class OSTrackONNX(nn.Module):
    """
    Wrapper for OSTrack that accepts template + search as separate inputs.
    OSTrack processes them together internally via token concatenation.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, template_img, search_img):
        # OSTrack internally concatenates template tokens + search tokens
        # and runs them through the shared ViT backbone
        out = self.model(template_img, search_img)
        
        # out contains 'pred_boxes' and optionally 'score_map'
        pred_boxes = out['pred_boxes']   # shape: (1, 4) — cx, cy, w, h normalized
        score_map = out['score_map']     # shape: (1, 1, H, W)
        
        return pred_boxes, score_map


def export_model(config_path, checkpoint_path, output_path="web/models/ostrack.onnx"):
    """Export OSTrack to a single ONNX file."""
    
    # --- Load the real model ---
    model = load_model(config_path, checkpoint_path)
    wrapped = OSTrackONNX(model)
    wrapped.eval()
    
    # --- Dummy inputs matching OSTrack's expected sizes ---
    # Template: 128x128 crop of the initial target
    # Search:   256x256 crop of the current frame region
    dummy_template = torch.randn(1, 3, 128, 128)
    dummy_search   = torch.randn(1, 3, 256, 256)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting ONNX to: {output_path}")
    
    torch.onnx.export(
        wrapped,
        (dummy_template, dummy_search),
        output_path,
        input_names=['template_img', 'search_img'],
        output_names=['pred_boxes', 'score_map'],
        dynamic_axes={
            'template_img': {0: 'batch'},
            'search_img':   {0: 'batch'},
            'pred_boxes':   {0: 'batch'},
            'score_map':    {0: 'batch'},
        },
        opset_version=13,
        do_constant_folding=True,   # Optimizes the graph for inference
        verbose=False
    )
    
    print("Export complete!")
    
    # --- Verify the exported model is valid ---
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified — no errors found.")
    
    # --- Optional: print file size ---
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")


if __name__ == "__main__":
    # Robust absolute paths to the OSTrack configuration and downloaded weights
    CONFIG_PATH     = os.path.join(ostrack_path, "experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml")
    CHECKPOINT_PATH = os.path.join(ostrack_path, "output/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar")
    
    # Example export output path (relative to the notebook/script location or absolute)
    OUTPUT_PATH = os.path.join(parent_dir, "web/models/ostrack.onnx")

    export_model(CONFIG_PATH, CHECKPOINT_PATH, output_path=OUTPUT_PATH)