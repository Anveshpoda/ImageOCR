import cv2

# Read the image
image = cv2.imread('test_signboard.jpg')

# Check if image was loaded successfully
if image is None:
    print("Error: Could not read the image file")
else:
    # Get image properties
    height, width, channels = image.shape
    print(f"Image loaded successfully")
    print(f"Dimensions: {width}x{height}")
    print(f"Number of channels: {channels}")
    print(f"Data type: {image.dtype}")
