import cv2
import numpy as np

# Read the image
image = cv2.imread('test_signboard.jpg')

if image is None:
    print("Error: Could not read the image file")
else:
    # Display the image
    cv2.imshow('Test Image', image)
    print("Displaying image. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print some pixel values
    print("\nTop-left corner pixel (BGR):", image[0, 0])
    print("Image mean value:", np.mean(image))
