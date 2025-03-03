from ultralytics import YOLO
import cv2

class VehicleDetector:
    def __init__(self, model_weights='yolov8m.pt', conf_threshold=0.4):
        # Load YOLO model for vehicle detection
        self.model = YOLO(model_weights)
        self.conf_threshold = conf_threshold

    def detect_vehicles(self, image_path):
        # Run YOLO detection on the image
        return self.model(image_path, conf=self.conf_threshold)

    def is_point_inside_box(self, point, box):
        # Check if a given point (x, y) lies inside a given box (x1, y1, x2, y2)
        x, y = point
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def detect_and_count_with_image(self, image_path):
        # Run detection
        results = self.detect_vehicles(image_path)

        # Classes we consider as "vehicles" (car, motorcycle, bus, truck)
        vehicle_classes = {2, 3, 5, 7}

        # Load image using OpenCV
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Define the lower half of the image as detection zone
        detection_zone = (0, height // 2, width, height)

        # Count vehicles in the detection zone
        vehicle_count = 0

        for result in results:
            for box in result.boxes:
                # Get class id for detected object
                object_class = int(box.cls)

                # Only count if it's a vehicle (car, bike, bus, truck)
                if object_class in vehicle_classes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Find the bottom-center point of the box
                    bottom_center = ((x1 + x2) // 2, y2)

                    # Check if bottom-center point is inside the lower half
                    if self.is_point_inside_box(bottom_center, detection_zone):
                        vehicle_count += 1

        # Draw a red line separating the upper and lower halves
        annotated_image = image.copy()
        cv2.rectangle(annotated_image, (0, height // 2), (width, height), (0, 0, 255), 2)

        # Convert BGR to RGB for display (if using matplotlib/streamlit)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Return vehicle count, annotated image, and full detection result (if needed later)
        return vehicle_count, annotated_image_rgb, results
