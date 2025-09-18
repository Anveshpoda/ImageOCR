import cv2
import numpy as np
import json
from paddleocr import PaddleOCR
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_ocr_result(result, title="OCR Result"):
    """Helper function to print OCR results in a readable format"""
    print(f"\n{title}:")
    print("=" * 80)
    print(json.dumps(result, indent=2, default=str))
    print("=" * 80)

def main():
    # Initialize PaddleOCR with minimal configuration
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(lang='en')
    print("PaddleOCR initialized successfully")
    
    # Test 1: Try with the test signboard image
    # test_image_path = 'test_signboard.jpg'
    test_image_path = 'test_img.jpg'
    print(f"\nTesting with image: {test_image_path}")
    
    # Read and display image info
    img = cv2.imread(test_image_path)
    if img is None:
        print("Error: Could not read the image file")
        return
    
    print(f"Image loaded - Size: {img.shape[1]}x{img.shape[0]}, Channels: {img.shape[2]}")
    print(f"Mean pixel value: {np.mean(img):.2f}")
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run OCR with default parameters
    print("\nRunning OCR...")
    try:
        # Use predict() directly as suggested by the deprecation warning
        result = ocr.predict(img_rgb)
        print_ocr_result(result, "OCR Result")
        
        # Process the result based on its actual structure
        if not result:
            print("No OCR results returned")
            return
            
        # Print the structure of the first result
        first_result = result[0] if isinstance(result, list) else result
        print(f"\nFirst result type: {type(first_result).__name__}")
        
        # Try to extract text from the result
        if hasattr(first_result, 'txt'):
            # Handle OCRResult object
            print(f"Detected text: {first_result.txt}")
            print(f"Confidence: {getattr(first_result, 'confidence', 'N/A')}")
        elif hasattr(first_result, '__dict__'):
            # Print all attributes of the result object
            print("Result object attributes:")
            for attr, value in first_result.__dict__.items():
                print(f"  {attr}: {value}")
        else:
            print("Unexpected result format. Raw result:")
            print_ocr_result(result, "Raw OCR Result")
            
    except Exception as e:
        print(f"\nError during OCR processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
