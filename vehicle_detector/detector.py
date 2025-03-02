from ultralytics import YOLO
import cv2
import os

class VehicleDetector:
    def __init__(self, model_weights='yolov8m.pt', conf_threshold=0.4):
        self.model = YOLO(model_weights)
        self.conf_threshold = conf_threshold

    def detect_vehicles(self, image_path):
        return self.model(image_path, conf=self.conf_threshold)

    def is_point_inside_box(self, point, box):
        x, y = point
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def detect_and_count_with_image(self, image_path):
        results = self.detect_vehicles(image_path)

        # COCO vehicle class IDs: car, motorcycle, bus, truck
        vehicle_classes = {2, 3, 5, 7}

        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        detection_zone = (0, h // 2, w, h)

        count = 0
        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                if cls in vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    bottom_center = ((x1 + x2) // 2, y2)

                    if self.is_point_inside_box(bottom_center, detection_zone):
                        count += 1

            # Draw all detections
            annotated_img = image.copy()

            # Draw the detection zone (bottom half box)
            cv2.rectangle(annotated_img, (detection_zone[0], detection_zone[1]),
                          (detection_zone[2], detection_zone[3]), (0, 0, 255), 2)
            cv2.putText(annotated_img, f'', (10, detection_zone[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return count, cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    detector = VehicleDetector()

    input_folder = os.path.join(os.path.dirname(__file__), '../data/input_images')
    output_folder = os.path.join(os.path.dirname(__file__), '../data/output_images')

    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"❌ Input folder '{input_folder}' not found! Add images to test.")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"🚦 Processing images from: {input_folder}")

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)

            # Run detection and count vehicles
            count, annotated_image = detector.detect_and_count_with_image(image_path)

            # Save annotated image to output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            print(f"✅ Processed '{filename}' - Vehicle Count: {count}")

    print("🎉 All images processed! Check 'data/output_images' for results.")
