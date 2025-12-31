import torch
import torch.nn as nn
from torchvision import models

# 1. Load the Model Structure
model = models.mobilenet_v2(weights=None) 
model.classifier[1] = nn.Linear(model.last_channel, 3) # 3 Classes

# 2. Load Trained Weights
model.load_state_dict(torch.load('robocon_model.pth', map_location='cpu'))
model.eval()

# 3. Create Dummy Input (1 batch, 3 channels, 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# 4. Export
onnx_path = "robocon_model.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    verbose=False,
    input_names=['input'], 
    output_names=['output'],
    opset_version=18
)

print(f"Model exported to {onnx_path}")