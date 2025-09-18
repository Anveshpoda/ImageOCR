import cv2
import numpy as np
from ocr import SignboardExtractor

# Initialize
extractor = SignboardExtractor()

# Load the test image
image_path = "test_signboard.jpg"
print(f"Processing image: {image_path}")

# Display image info
img = cv2.imread(image_path)
if img is None:
    print("Error: Could not read the image file")
    exit(1)

print(f"\nImage loaded successfully")
print(f"Dimensions: {img.shape[1]}x{img.shape[0]}")
print(f"Channels: {img.shape[2] if len(img.shape) > 2 else 1}")
print(f"Data type: {img.dtype}")

# Save a copy of the image for verification
cv2.imwrite("debug_original.jpg", img)
print("\nSaved original image as: debug_original.jpg")

# Test OCR
print("\nRunning OCR...")
full_text, text_blocks = extractor.extract_text_with_positions(image_path)

print("\nFull extracted text:")
print(f'"{full_text}"')

print("\nText blocks with positions:")
for i, block in enumerate(text_blocks, 1):
    print(f"\nBlock {i}:")
    print(f"Text: {block['text']}")
    print(f"Confidence: {block['confidence']:.2f}")
    print(f"Position: {block['position']}")

# Draw bounding boxes on the image for debugging
debug_img = img.copy()
for block in text_blocks:
    bbox = np.array(block['bbox'], dtype=np.int32)
    cv2.polylines(debug_img, [bbox], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(debug_img, block['text'], (int(block['position']['left']), int(block['position']['top'] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Save the debug image
debug_image_path = "debug_ocr_result.jpg"
cv2.imwrite(debug_image_path, debug_img)
print(f"\nSaved debug image with OCR results as: {debug_image_path}")

# Extract structured data
print("\nExtracting structured data...")
result = extractor.extract_data(image_path)

print("\nExtracted Information:")
for key, value in result.items():
    print(f"{key.capitalize()}: {value}")

print("\nTest complete.")