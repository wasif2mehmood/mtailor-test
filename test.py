import os
from model import ImageClassifier

def simple_inference_test():
    """Simple test to load images and run inference"""
    
    print("=== Simple Image Classification Test ===\n")
    
    # Check if ONNX model exists
    if not os.path.exists("model.onnx"):
        print("model.onnx not found!")
        print("Please run: python convert_to_onnx.py first")
        return
    
    # Initialize classifier
    try:
        print("Loading ONNX model...")
        classifier = ImageClassifier("model.onnx")
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test images to classify
    test_images = [
        "n01440764_tench.JPEG",
        "n01667114_mud_turtle.JPEG"
    ]
    
    # Run inference on each image
    for image_path in test_images:
        print(f"Testing image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            print(f" Please make sure {image_path} is in the current directory\n")
            continue
        
        try:
            # Run classification
            result = classifier.classify_image(image_path)
            
            # Print results
            print(f" Classification successful!")
            print(f"   Class ID: {result['class_id']}")
            print(f"   Class Name: {result['class_name']}")
            print(f"   Confidence: {result['confidence']:.4f}")
            
            # Expected results check
            if image_path == "n01440764_tench.JPEG":
                expected_class = 0
                status = " CORRECT" if result['class_id'] == expected_class else "❌ INCORRECT"
                print(f"   Expected class: {expected_class} - {status}")
            elif image_path == "n01667114_mud_turtle.JPEG":
                expected_class = 35
                status = " CORRECT" if result['class_id'] == expected_class else "❌ INCORRECT"
                print(f"   Expected class: {expected_class} - {status}")
            
            print()
            
        except Exception as e:
            print(f"Error classifying {image_path}: {e}\n")

if __name__ == "__main__":
    simple_inference_test()