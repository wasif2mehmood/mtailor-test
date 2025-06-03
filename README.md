# ONNX Image Classifier - Cerebrium Deployment

A FastAPI-based image classification service using ONNX models, deployable on Cerebrium cloud platform.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Cerebrium account
- ONNX model file (`model.onnx`)
- Test images

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Login to Cerebrium
```bash
cerebrium login
```
Follow the prompts to authenticate with your Cerebrium account.

### 3. Deploy to Cerebrium
```bash
python -m cerebrium deploy --disable-syntax-check
```

The `--disable-syntax-check` flag bypasses linting issues during deployment.

### 4. Test Your Deployment
```bash
python test_server.py
```

## 📁 Project Structure

```
mtailor-test/
├── main.py                 # FastAPI application
├── model.py                # Image classifier wrapper
├── model.onnx              # ONNX model file
├── test_server.py          # Comprehensive test suite
├── requirements.txt        # Python dependencies
├── n01440764_tench.JPEG    # Test image 1
├── n01667114_mud_turtle.JPEG # Test image 2
└── README.md              # This file
```

## 🛠️ Core Files

### `main.py`
FastAPI application with endpoints:
- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /model-info` - Model information
- `POST /predict` - Single image prediction
- `POST /predict-top-k` - Top-k predictions

### `model.py`
ONNX model wrapper with image preprocessing and classification methods.

### `test_server.py`
Comprehensive test suite that validates:
- ✅ Model file existence
- ✅ Test image validation
- ✅ Model loading performance
- ✅ Basic inference accuracy
- ✅ Top-k predictions
- ✅ Edge case handling
- ✅ Performance benchmarks
- ✅ Cerebrium API endpoints

## 📋 Requirements

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
onnxruntime==1.16.3
numpy==1.24.3
Pillow==10.1.0
opencv-python-headless==4.8.1.78
python-multipart==0.0.6
requests==2.31.0
```

## 🔧 Configuration

### Cerebrium Configuration (Optional)
Create `cerebrium.toml` for custom settings:

```toml
[cerebrium.deployment]
name = "onnx-image-classifier"
python_version = "3.9"
region = "us-east-1"

[cerebrium.hardware]
cpu = 2
memory = 8.0
gpu_count = 0

[cerebrium.dependencies]
pip = [
    "fastapi",
    "uvicorn[standard]",
    "onnxruntime",
    "numpy",
    "Pillow",
    "opencv-python-headless",
    "python-multipart"
]
```

## 🧪 Testing

### Run Complete Test Suite
```bash
python test_server.py
```

This will test:
1. **Model Files Validation** - Checks if required files exist
2. **Test Images Validation** - Verifies test images are available
3. **Model Loading Test** - Tests ONNX model initialization
4. **Basic Inference Test** - Validates prediction accuracy
5. **Top-K Predictions Test** - Tests multiple predictions
6. **Edge Cases & Error Handling** - Tests error scenarios
7. **Performance Benchmarks** - Measures inference speed
8. **Cerebrium API Test** - Validates deployed endpoints

### Expected Test Results
```
🎯 Expected Predictions:
- n01440764_tench.JPEG → Class 0 (tench)
- n01667114_mud_turtle.JPEG → Class 35 (mud turtle)
```

## 📊 API Usage

### Health Check
```bash
curl -X GET "https://api.cortex.cerebrium.ai/v4/YOUR-PROJECT/mtailor-test/health" \
  -H "Authorization: Bearer YOUR-TOKEN"
```

### Single Prediction
```bash
curl -X POST "https://api.cortex.cerebrium.ai/v4/YOUR-PROJECT/mtailor-test/predict" \
  -H "Authorization: Bearer YOUR-TOKEN" \
  -F "file=@image.jpg"
```

### Top-K Predictions
```bash
curl -X POST "https://api.cortex.cerebrium.ai/v4/YOUR-PROJECT/mtailor-test/predict-top-k?k=5" \
  -H "Authorization: Bearer YOUR-TOKEN" \
  -F "file=@image.jpg"
```

## 🔍 Troubleshooting

### Common Issues

**1. Model Loading Failed**
```
❌ Model loading failed: [Errno 2] No such file or directory: 'model.onnx'
```
**Solution:** Ensure `model.onnx` is in the project root directory.

**2. Missing Dependencies**
```
❌ RuntimeError: Form data requires "python-multipart" to be installed.
```
**Solution:** Add `python-multipart` to `requirements.txt`.

**3. Authentication Error**
```
❌ Health endpoint failed: 401
```
**Solution:** Run `cerebrium login` and ensure valid authentication.

**4. Deployment Timeout**
```
❌ Deployment failed: timeout
```
**Solution:** Use `--disable-syntax-check` flag and check file sizes.

### Debug Commands

**Check Cerebrium Status:**
```bash
cerebrium status
```

**View Deployment Logs:**
```bash
cerebrium logs
```

**Redeploy with Force:**
```bash
cerebrium deploy --force --disable-syntax-check
```

## 📈 Performance Expectations

- **Model Loading:** < 3 seconds
- **Inference Time:** < 1 second per image
- **API Response:** < 2 seconds end-to-end
- **Memory Usage:** < 1GB
- **Model Size:** < 100MB (recommended)

## 🎯 Production Checklist

- [ ] All tests pass (8/8)
- [ ] Model accuracy validated
- [ ] Performance benchmarks meet requirements
- [ ] Error handling tested
- [ ] API endpoints functional
- [ ] Authentication working
- [ ] Monitoring configured
- [ ] Documentation complete

## 🔄 Development Workflow

1. **Develop Locally**
   ```bash
   python test_server.py  # Test local model
   ```

2. **Deploy to Cerebrium**
   ```bash
   cerebrium login
   python -m cerebrium deploy --disable-syntax-check
   ```

3. **Validate Deployment**
   ```bash
   python test_server.py  # Test deployed API
   ```

4. **Monitor Performance**
   ```bash
   cerebrium logs
   cerebrium status
   ```

## 📞 Support

For issues:
1. Check troubleshooting section
2. Review test output for specific failures
3. Check Cerebrium logs: `cerebrium logs`
4. Verify model file integrity
5. Ensure all dependencies are installed

## 🎉 Success Indicators

When everything works correctly, you should see:

```
🎉 ALL TESTS PASSED! Your model deployment is ready for production!

📊 Overall Results: 8/8 tests passed (100.0%)

🔍 LOCAL MODEL INFERENCE RESULTS:
✅ Correct predictions on test images
⚡ Fast inference times (< 1s)

🌐 CEREBRIUM API INFERENCE RESULTS:
✅ API endpoints responding
✅ Predictions matching local results
```

Your ONNX image classifier is now deployed and ready for production use! 🚀