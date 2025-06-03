import os
import time
import json
import requests
from pathlib import Path
from model import ImageClassifier
import numpy as np
from PIL import Image

class MLModelDeploymentTester:
    """Comprehensive test suite for ML model deployment"""
    
    def __init__(self):
        self.model_path = "model.onnx"
        self.test_images = [
            "n01440764_tench.JPEG",
            "n01667114_mud_turtle.JPEG"
        ]
        self.expected_results = {
            "n01440764_tench.JPEG": {"class_id": 0, "class_name": "tench"},
            "n01667114_mud_turtle.JPEG": {"class_id": 35, "class_name": "mud turtle"}
        }
        self.classifier = None
        self.inference_results = []  # Store inference results
        self.cerebrium_results = []  # Store Cerebrium API results
        
        # Cerebrium API configuration
        self.cerebrium_url = "https://api.cortex.cerebrium.ai/v4/p-fd5b3362/mtailor-test"
        self.cerebrium_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWZkNWIzMzYyIiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoyMDY0NDY2MDYxfQ.Vz1zPvecd1W5Ruzh03JDi39cfBxOhEbmGl_dHfpbn5y_sV3-XbpV3IR_3jeRxGADX0q9sIOE17XoPtwtu1bR_E3YAAVvQCdp65M02mc70lEhGv_udJbJFUhGSQZEO-msvZ6Z9jjz4BfcpSKZ3Xwm0bGpnhHAi1qPgE3oSEeZ1N0YVK32qoMk6GW-D1X_RaMD1GIO3_VANZb2VCjrlAhc5W8lko6gDlJj7V9ReBeAEa6GKMvsNUdcevu_S3_otfG_Sa3DC32qdMr4CoXF7-me5qLEyHkIpOO3fc1dQQm7iaIwu28rCQRd39DKJVHlB1OAK1feF2sAZSV2llJRUneJzw"

    def print_section(self, title):
        """Print section header"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    def test_model_files_exist(self):
        """Test 1: Check if required model files exist"""
        self.print_section("1. MODEL FILES VALIDATION")
        
        required_files = [self.model_path, "model.py"]
        missing_files = []
        
        for file in required_files:
            if os.path.exists(file):
                size = os.path.getsize(file) / (1024*1024)  # Size in MB
                print(f"✅ {file} exists ({size:.2f} MB)")
            else:
                missing_files.append(file)
                print(f"❌ {file} missing")
        
        return len(missing_files) == 0

    def test_test_images_exist(self):
        """Test 2: Check if test images exist"""
        self.print_section("2. TEST IMAGES VALIDATION")
        
        missing_images = []
        for image in self.test_images:
            if os.path.exists(image):
                # Get image info
                with Image.open(image) as img:
                    print(f"✅ {image} exists ({img.size[0]}x{img.size[1]}, {img.mode})")
            else:
                missing_images.append(image)
                print(f"❌ {image} missing")
        
        return len(missing_images) == 0

    def test_model_loading(self):
        """Test 3: Model loading and initialization"""
        self.print_section("3. MODEL LOADING TEST")
        
        try:
            start_time = time.time()
            self.classifier = ImageClassifier(self.model_path)
            load_time = time.time() - start_time
            
            print(f"✅ Model loaded successfully in {load_time:.3f} seconds")
            print(f"   Input name: {self.classifier.model.input_name}")
            print(f"   Output name: {self.classifier.model.output_name}")
            print(f"   Available classes: {len(self.classifier.class_names)}")
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False

    def test_basic_inference(self):
        """Test 4: Basic inference functionality"""
        self.print_section("4. BASIC INFERENCE TEST")
        
        if not self.classifier:
            print("❌ Model not loaded, skipping inference test")
            return False
        
        success_count = 0
        total_time = 0
        
        for image_path in self.test_images:
            if not os.path.exists(image_path):
                print(f"❌ {image_path} not found, skipping")
                continue
                
            try:
                print(f"\nTesting: {image_path}")
                start_time = time.time()
                result = self.classifier.classify_image(image_path)
                inference_time = time.time() - start_time
                total_time += inference_time
                
                # Store result for final summary
                self.inference_results.append({
                    "image": image_path,
                    "result": result,
                    "inference_time": inference_time
                })
                
                print(f"  ✅ Inference successful ({inference_time:.3f}s)")
                print(f"     Class ID: {result['class_id']}")
                print(f"     Class Name: {result['class_name']}")
                print(f"     Confidence: {result['confidence']:.4f}")
                
                # Check expected results
                if image_path in self.expected_results:
                    expected = self.expected_results[image_path]
                    if result['class_id'] == expected['class_id']:
                        print(f"     ✅ Correct prediction!")
                    else:
                        print(f"     ⚠️  Expected class {expected['class_id']}, got {result['class_id']}")
                
                success_count += 1
                
            except Exception as e:
                print(f"  ❌ Inference failed: {e}")
        
        avg_time = total_time / success_count if success_count > 0 else 0
        print(f"\n📊 Inference Summary: {success_count}/{len(self.test_images)} successful")
        print(f"   Average inference time: {avg_time:.3f}s")
        
        return success_count == len(self.test_images)

    def test_top_k_predictions(self):
        """Test 5: Top-k predictions"""
        self.print_section("5. TOP-K PREDICTIONS TEST")
        
        if not self.classifier:
            print("❌ Model not loaded, skipping top-k test")
            return False
        
        if not os.path.exists(self.test_images[0]):
            print(f"❌ Test image {self.test_images[0]} not found")
            return False
        
        try:
            k_values = [3, 5]
            for k in k_values:
                print(f"\nTesting top-{k} predictions:")
                results = self.classifier.get_top_k_predictions(self.test_images[0], k)
                
                print(f"  ✅ Got {len(results)} predictions")
                for i, pred in enumerate(results, 1):
                    print(f"     {i}. {pred['class_name']} ({pred['confidence']:.4f})")
                
                # Validate results
                if len(results) == k:
                    print(f"  ✅ Correct number of predictions returned")
                else:
                    print(f"  ❌ Expected {k} predictions, got {len(results)}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Top-k predictions failed: {e}")
            return False

    def test_edge_cases(self):
        """Test 6: Edge cases and error handling"""
        self.print_section("6. EDGE CASES & ERROR HANDLING")
        
        if not self.classifier:
            print("❌ Model not loaded, skipping edge case tests")
            return False
        
        test_cases = [
            ("Non-existent file", "nonexistent.jpg"),
            ("Invalid image format", self._create_invalid_image()),
            ("Empty file", self._create_empty_file())
        ]
        
        passed_tests = 0
        
        for test_name, test_file in test_cases:
            print(f"\nTesting: {test_name}")
            try:
                if test_file and os.path.exists(test_file):
                    result = self.classifier.classify_image(test_file)
                    print(f"  ⚠️  Expected failure but got result: {result}")
                else:
                    result = self.classifier.classify_image(test_file)
                    print(f"  ⚠️  Expected failure but got result: {result}")
            except Exception as e:
                print(f"  ✅ Properly handled error: {type(e).__name__}")
                passed_tests += 1
        
        # Cleanup temporary files
        for _, test_file in test_cases:
            if test_file and os.path.exists(test_file) and test_file.startswith("temp_"):
                os.remove(test_file)
        
        return passed_tests >= 2  # At least 2/3 edge cases should be handled

    def test_performance(self):
        """Test 7: Performance benchmarks"""
        self.print_section("7. PERFORMANCE BENCHMARKS")
        
        if not self.classifier or not os.path.exists(self.test_images[0]):
            print("❌ Skipping performance test")
            return False
        
        try:
            # Warm-up run
            self.classifier.classify_image(self.test_images[0])
            
            # Performance test
            num_runs = 10
            times = []
            
            print(f"Running {num_runs} inference iterations...")
            for i in range(num_runs):
                start_time = time.time()
                self.classifier.classify_image(self.test_images[0])
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)
            
            print(f"📊 Performance Results:")
            print(f"   Average: {avg_time:.3f}s")
            print(f"   Min: {min_time:.3f}s")
            print(f"   Max: {max_time:.3f}s")
            print(f"   Std Dev: {std_time:.3f}s")
            
            # Performance thresholds
            if avg_time < 1.0:
                print(f"   ✅ Performance: Excellent (< 1s)")
            elif avg_time < 3.0:
                print(f"   ✅ Performance: Good (< 3s)")
            else:
                print(f"   ⚠️  Performance: Needs optimization (> 3s)")
            
            return True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return False

    def test_cerebrium_deployment(self):
        """Test 8: Cerebrium deployment"""
        self.print_section("8. CEREBRIUM API DEPLOYMENT TEST")
        
        headers = {'Authorization': f'Bearer {self.cerebrium_token}'}
        
        # Test health endpoint
        try:
            response = requests.get(f"{self.cerebrium_url}/health", headers=headers, timeout=30)
            if response.status_code == 200:
                print("✅ Health endpoint working")
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health endpoint error: {e}")
            return False
        
        # Test model info endpoint
        try:
            response = requests.get(f"{self.cerebrium_url}/model-info", headers=headers, timeout=30)
            if response.status_code == 200:
                print("✅ Model info endpoint working")
                info = response.json()
                print(f"   Model: {info.get('model_path', 'N/A')}")
                print(f"   Classes: {info.get('available_classes', 'N/A')}")
            else:
                print(f"❌ Model info endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️  Model info endpoint error: {e}")
        
        # Test prediction endpoint
        if not os.path.exists(self.test_images[0]):
            print(f"❌ Test image not found, skipping prediction test")
            return False
        
        try:
            with open(self.test_images[0], "rb") as f:
                files = {"file": (self.test_images[0], f, "image/jpeg")}
                response = requests.post(f"{self.cerebrium_url}/predict", 
                                       headers=headers, files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                # Store Cerebrium result for final summary
                self.cerebrium_results.append({
                    "image": self.test_images[0],
                    "result": result
                })
                
                print("✅ Prediction endpoint working")
                print(f"   Predicted: {result.get('class_name', 'N/A')} ({result.get('confidence', 0):.4f})")
                
                # Test top-k endpoint
                try:
                    with open(self.test_images[0], "rb") as f:
                        files = {"file": (self.test_images[0], f, "image/jpeg")}
                        params = {"k": 3}
                        response = requests.post(f"{self.cerebrium_url}/predict-top-k", 
                                               headers=headers, files=files, params=params, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        print("✅ Top-k prediction endpoint working")
                        print(f"   Got {result.get('count', 0)} predictions")
                    else:
                        print(f"⚠️  Top-k endpoint failed: {response.status_code}")
                except Exception as e:
                    print(f"⚠️  Top-k endpoint error: {e}")
                
                return True
            else:
                print(f"❌ Prediction endpoint failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Prediction endpoint error: {e}")
            return False

    def _create_invalid_image(self):
        """Create an invalid image file for testing"""
        filename = "temp_invalid.jpg"
        with open(filename, "w") as f:
            f.write("This is not an image file")
        return filename

    def _create_empty_file(self):
        """Create an empty file for testing"""
        filename = "temp_empty.jpg"
        open(filename, "w").close()
        return filename

    def print_inference_outputs(self):
        """Print all inference outputs in a summary"""
        self.print_section("INFERENCE OUTPUTS SUMMARY")
        
        if self.inference_results:
            print("🔍 LOCAL MODEL INFERENCE RESULTS:")
            print("-" * 50)
            
            for i, result_data in enumerate(self.inference_results, 1):
                image = result_data["image"]
                result = result_data["result"]
                inference_time = result_data["inference_time"]
                
                print(f"\n{i}. Image: {image}")
                print(f"   📊 Result:")
                print(f"      Class ID: {result['class_id']}")
                print(f"      Class Name: {result['class_name']}")
                print(f"      Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
                print(f"      Image Path: {result.get('image_path', 'N/A')}")
                print(f"      Inference Time: {inference_time:.3f}s")
                
                # Show expected vs actual
                if image in self.expected_results:
                    expected = self.expected_results[image]
                    status = "✅ CORRECT" if result['class_id'] == expected['class_id'] else "❌ INCORRECT"
                    print(f"      Expected: Class {expected['class_id']} ({expected['class_name']}) - {status}")
                
                print(f"   📄 Full JSON Response:")
                print(f"      {json.dumps(result, indent=6)}")
        
        if self.cerebrium_results:
            print("\n🌐 CEREBRIUM API INFERENCE RESULTS:")
            print("-" * 50)
            
            for i, result_data in enumerate(self.cerebrium_results, 1):
                image = result_data["image"]
                result = result_data["result"]
                
                print(f"\n{i}. Image: {image}")
                print(f"   📊 Cerebrium Result:")
                print(f"      Class ID: {result.get('class_id', 'N/A')}")
                print(f"      Class Name: {result.get('class_name', 'N/A')}")
                print(f"      Confidence: {result.get('confidence', 0):.4f} ({result.get('confidence', 0)*100:.2f}%)")
                print(f"      Image Path: {result.get('image_path', 'N/A')}")
                
                print(f"   📄 Full JSON Response:")
                print(f"      {json.dumps(result, indent=6)}")

    def run_all_tests(self):
        """Run all tests"""
        print("🚀 COMPREHENSIVE ML MODEL DEPLOYMENT TEST SUITE")
        print("Testing core functionality and Cerebrium deployment")
        
        test_results = []
        
        # Core functionality tests
        test_results.append(("Model Files", self.test_model_files_exist()))
        test_results.append(("Test Images", self.test_test_images_exist()))
        test_results.append(("Model Loading", self.test_model_loading()))
        test_results.append(("Basic Inference", self.test_basic_inference()))
        test_results.append(("Top-K Predictions", self.test_top_k_predictions()))
        test_results.append(("Edge Cases", self.test_edge_cases()))
        test_results.append(("Performance", self.test_performance()))
        
        # Deployment test
        test_results.append(("Cerebrium API", self.test_cerebrium_deployment()))
        
        # Print final results
        self.print_section("FINAL TEST RESULTS")
        
        passed = 0
        total = 0
        
        for test_name, result in test_results:
            if result == "SKIPPED":
                print(f"⏭️  {test_name:<20} SKIPPED")
            elif result:
                print(f"✅ {test_name:<20} PASSED")
                passed += 1
                total += 1
            else:
                print(f"❌ {test_name:<20} FAILED")
                total += 1
        
        print(f"\n📊 Overall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Your model deployment is ready for production!")
        elif passed >= total * 0.8:
            print("✅ Most tests passed. Your deployment is mostly ready!")
        else:
            print("⚠️  Several tests failed. Please review and fix issues before deployment.")
        
        # Print all inference outputs at the end
        self.print_inference_outputs()
        
        return passed == total

if __name__ == "__main__":
    tester = MLModelDeploymentTester()
    tester.run_all_tests()