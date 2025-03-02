import os
import random
from vehicle_detector.detector import VehicleDetector

class TrafficSignalController:
    def __init__(self, model_name="yolov8m"):
        self.detector = VehicleDetector(model_weights=f'{model_name}.pt', conf_threshold=0.4)
        self.image_folder = os.path.join(os.path.dirname(__file__), 'data/output_images')

    def pick_random_images(self, count=4):
        all_images = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if len(all_images) < count:
            raise FileNotFoundError(f"Not enough images found! Expected {count}, found {len(all_images)}.")
        return random.sample(all_images, count)

    def calculate_vehicle_counts_with_images(self):
        random_images = self.pick_random_images()
        counts = {}
        annotated_images = []

        for idx, image_name in enumerate(random_images):
            image_path = os.path.join(self.image_folder, image_name)
            count, annotated_image = self.detector.detect_and_count_with_image(image_path)
            counts[f"Direction_{idx + 1}"] = count
            annotated_images.append(annotated_image)

        return counts, annotated_images

    def decide_signal_timing(self, counts):
        max_count = max(counts.values())
        base_time = 10  # Minimum green time in seconds

        timings = {}
        for direction, count in counts.items():
            if max_count == 0:
                timings[direction] = base_time
            else:
                timings[direction] = base_time + int((count / max_count) * 20)

        return timings

    def run_control_cycle(self):
        counts, annotated_images = self.calculate_vehicle_counts_with_images()
        timings = self.decide_signal_timing(counts)
        return counts, timings, annotated_images

if __name__ == "__main__":
    controller = TrafficSignalController()
    controller.run_control_cycle()
