import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
from typing import Union, Tuple, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class ImagePreprocessor:
    """
    Image preprocessing class for ImageNet model
    """
    
    def __init__(self):
        self.target_size = (224, 224)
        # IMPORTANT: Set dtype explicitly for all numpy arrays
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for model inference
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.info("Converted image to RGB format")
            
            # Resize to 224x224 using bilinear interpolation
            image = image.resize(self.target_size, Image.BILINEAR)
            
            # Convert to numpy array and normalize to [0, 1]
            # CRITICAL: Ensure float32 from the start
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0
            
            # Normalize using ImageNet statistics
            # CRITICAL: Ensure all operations maintain float32
            image_array = (image_array - self.mean) / self.std
            
            # Convert from HWC to CHW format
            image_array = np.transpose(image_array, (2, 0, 1))
            
            # Add batch dimension and FORCE float32
            image_array = np.expand_dims(image_array, axis=0)
            image_array = image_array.astype(np.float32)  # Force conversion
            
            # Debug: Print actual dtype
            print(f"DEBUG: Final tensor dtype: {image_array.dtype}")
            logger.info(f"Image preprocessed successfully. Shape: {image_array.shape}, dtype: {image_array.dtype}")
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
class ONNXModel:
    """
    ONNX Model loading and prediction class
    """
    
    def __init__(self, model_path: str = "model.onnx"):
        """
        Initialize ONNX model
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.load_model()
    
    def load_model(self):
        """Load ONNX model and initialize session"""
        try:
            # Create ONNX Runtime session
            self.session = ort.InferenceSession(self.model_path)
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"✓ ONNX model loaded successfully from {self.model_path}")
            logger.info(f"Input name: {self.input_name}")
            logger.info(f"Output name: {self.output_name}")
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            raise
    
    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference on input tensor
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            Model predictions (logits)
        """
        try:
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            
            logger.info(f"Inference completed. Output shape: {outputs[0].shape}")
            return outputs[0]
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise
    
    def predict_class(self, input_tensor: np.ndarray) -> Tuple[int, float]:
        """
        Predict class and confidence
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            Tuple of (predicted_class_id, confidence_score)
        """
        predictions = self.predict(input_tensor)
        
        # Apply softmax to get probabilities
        probabilities = self._softmax(predictions[0])
        
        # Get predicted class
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return int(predicted_class), float(confidence)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax activation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


class ImageClassifier:
    """
    Complete image classification pipeline
    """
    
    def __init__(self, model_path: str = "model.onnx"):
        """
        Initialize the complete classification pipeline
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.preprocessor = ImagePreprocessor()
        self.model = ONNXModel(model_path)
        
        # ImageNet class mapping (sample - you might want to load full mapping)
        self.class_names = {
            0: "tench",
            35: "mud turtle",
            # Add more class mappings as needed
        }
    
    def classify_image(self, image_path: str) -> dict:
        """
        Complete image classification pipeline
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Classification results dictionary
        """
        try:
            # Preprocess image
            input_tensor = self.preprocessor.preprocess(image_path)
            
            # Run inference
            class_id, confidence = self.model.predict_class(input_tensor)
            
            # Get class name if available
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            result = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "image_path": image_path
            }
            
            logger.info(f"Classification result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in classification pipeline: {str(e)}")
            raise
    
    def get_top_k_predictions(self, image_path: str, k: int = 5) -> List[dict]:
        """
        Get top-k predictions
        
        Args:
            image_path: Path to the input image
            k: Number of top predictions to return
            
        Returns:
            List of top-k predictions
        """
        try:
            # Preprocess image
            input_tensor = self.preprocessor.preprocess(image_path)
            
            # Run inference
            predictions = self.model.predict(input_tensor)
            probabilities = self.model._softmax(predictions[0])
            
            # Get top-k indices
            top_k_indices = np.argsort(probabilities)[-k:][::-1]
            
            results = []
            for idx in top_k_indices:
                class_name = self.class_names.get(idx, f"class_{idx}")
                results.append({
                    "class_id": int(idx),
                    "class_name": class_name,
                    "confidence": float(probabilities[idx])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting top-k predictions: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    try:
        classifier = ImageClassifier("model.onnx")
        
        # Test with sample images
        if os.path.exists("n01440764_tench.JPEG"):
            result = classifier.classify_image("n01440764_tench.JPEG")
            print(f"Tench classification: {result}")
        
        if os.path.exists("n01667114_mud_turtle.JPEG"):
            result = classifier.classify_image("n01667114_mud_turtle.JPEG")
            print(f"Mud turtle classification: {result}")
            
    except Exception as e:
        print(f"Error: {e}")