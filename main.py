from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from model import ImageClassifier
import tempfile
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize the classifier globally (loads once at startup)
classifier = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global classifier
    try:
        logger.info("Loading ONNX model...")
        classifier = ImageClassifier("model.onnx")
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict image class using the ONNX model
    
    Args:
        file: Uploaded image file
        
    Returns:
        Classification results
    """
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Create temporary file to save uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Run classification
        result = classifier.classify_image(temp_file_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        # Clean up temporary file in case of error
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-top-k")
async def predict_top_k(file: UploadFile = File(...), k: int = 5):
    """
    Get top-k predictions for uploaded image
    
    Args:
        file: Uploaded image file
        k: Number of top predictions to return
        
    Returns:
        Top-k classification results
    """
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Create temporary file to save uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Run top-k classification
        results = classifier.get_top_k_predictions(temp_file_path, k)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return JSONResponse(content={"predictions": results, "count": len(results)})
        
    except Exception as e:
        # Clean up temporary file in case of error
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        logger.error(f"Top-k prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/hello")
def hello():
    return {"message": "Hello Cerebrium!"}

@app.get("/health")
def health():
    """Health check endpoint"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.get("/ready")
def ready():
    """Readiness check endpoint"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready", "model_loaded": True}

@app.get("/model-info")
def model_info():
    """Get model information"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": classifier.model.model_path,
        "input_name": classifier.model.input_name,
        "output_name": classifier.model.output_name,
        "available_classes": len(classifier.class_names),
        "class_names": classifier.class_names
    }