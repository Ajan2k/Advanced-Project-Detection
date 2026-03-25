import torch
import torch.nn as nn
import os

# Create the models directory locally if it doesn't exist
os.makedirs('../web/models', exist_ok=True)

class DummyTemplateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # A tiny network
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)

    def forward(self, template_img):
        # Outputs a dummy embedding [1, 16, 64, 64]
        return self.conv(template_img)

class DummyTrackingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Tiny components that do nothing useful but run fast
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(16 * 128 * 128 + 16 * 64 * 64, 4)
        
    def forward(self, search_img, template_features):
        x = self.conv(search_img) # [1, 16, 128, 128]
        # Flatten and concat just to make the graph use both inputs
        x_flat = x.view(1, -1)
        t_flat = template_features.view(1, -1)
        combined = torch.cat([x_flat, t_flat], dim=1)
        
        # Predict bounding box and score
        box = self.fc(combined)
        # Bounding box should be roughly centered [0.5, 0.5, 0.2, 0.2] or similar
        # We'll just sigmoid it so it's between 0 and 1
        box = torch.sigmoid(box)
        
        score = torch.tensor([1.0], dtype=torch.float32)
        return box, score

print("Exporting dummy Template Model...")
template_net = DummyTemplateEncoder()
dummy_template = torch.randn(1, 3, 128, 128)
torch.onnx.export(
    template_net, dummy_template, "../web/models/template_model.onnx",
    input_names=['template_img'], output_names=['template_feat'],
    opset_version=13
)

print("Exporting dummy Tracking Model...")
tracking_net = DummyTrackingNetwork()
dummy_search = torch.randn(1, 3, 256, 256)
dummy_z_feat = template_net(dummy_template)
torch.onnx.export(
    tracking_net, (dummy_search, dummy_z_feat), "../web/models/tracking_model.onnx",
    input_names=['search_img', 'template_feat'], 
    output_names=['pred_box', 'score'],
    opset_version=13
)

print("SUCCESS! Dummy models generated in web/models/")
