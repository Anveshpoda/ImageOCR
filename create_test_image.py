import cv2
import numpy as np

# Create a white image
image = np.ones((500, 800, 3), dtype=np.uint8) * 255

# Add some sample text
cv2.putText(image, 'Business Name: Sample Store', (50, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.putText(image, 'Mobile: +91 9876543210', (50, 150), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.putText(image, 'GST: 22ABCDE1234F1Z5', (50, 200), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.putText(image, 'Address: 123 Business Street, City', (50, 250), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.putText(image, 'Pincode: 400001', (50, 300), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Save the image
cv2.imwrite('test_signboard.jpg', image)
print("Test image created: test_signboard.jpg")
