import torch
import torch.onnx
from pytorch_model import Classifier, BasicBlock
import numpy as np

def convert_pytorch_to_onnx():
    """
    Convert PyTorch ResNet-18 model to ONNX format
    """
    # Load the PyTorch model
    model = Classifier(BasicBlock, [2, 2, 2, 2], num_classes=1000)
    
    # Load the weights (download from the provided link first)
    try:
        model.load_state_dict(torch.load("./pytorch_model_weights.pth", map_location='cpu'))
        print("✓ Model weights loaded successfully")
    except FileNotFoundError:
        print("❌ Model weights file not found. Please download from:")
        print("https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0")
        return False
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input tensor (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    torch.onnx.export(
        model,                          # PyTorch model
        dummy_input,                    # Input tensor
        "model.onnx",                   # Output file name
        export_params=True,             # Store trained parameter weights
        opset_version=11,               # ONNX version
        do_constant_folding=True,       # Optimize constant folding
        input_names=['input'],          # Input tensor name
        output_names=['output'],        # Output tensor name
        dynamic_axes={
            'input': {0: 'batch_size'},     # Variable batch size
            'output': {0: 'batch_size'}
        }
    )
    
    print("✓ Model successfully converted to ONNX format: model.onnx")
    
    # Verify the conversion
    import onnx
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verification passed")
    
    return True

if __name__ == "__main__":
    convert_pytorch_to_onnx()