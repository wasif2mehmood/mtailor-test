import os

# Get the project root directory (where this file is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model.onnx")

# Test data paths
TEST_IMAGES = {
    "tench": os.path.join(PROJECT_ROOT, "images", "n01440764_tench.jpeg"),
    "turtle": os.path.join(PROJECT_ROOT, "images", "n01667114_mud_turtle.JPEG")
}

# Expected classifications
EXPECTED_CLASSES = {
    "n01440764_tench.JPEG": 0,
    "n01667114_mud_turtle.JPEG": 35
}