from vehicle_detector.detector import VehicleDetector
import os
import cv2

# Initialize detector
detector = VehicleDetector()

# Use a test image (make sure this exists in the correct folder)
image_path = '../data/images/4_mp4-2_jpg.rf.085d0c1923f20d8253a0c88c2c9c9452.jpg'

# Run detection
count, annotated_image, _ = detector.detect_and_count_with_image(image_path)

# Show result
print(f"Detected vehicles: {count}")
cv2.imshow("Annotated Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
