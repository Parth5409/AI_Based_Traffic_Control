import os
import random
from vehicle_detector.detector import VehicleDetector

class TrafficSignalController:
    def __init__(self, model_name="yolov8m"):
        # Create a vehicle detector using the specified YOLO model
        self.detector = VehicleDetector(model_weights=f'{model_name}.pt', conf_threshold=0.4)

        # Folder where images are stored
        self.image_folder = os.path.join(os.path.dirname(__file__), 'data/images')

    def pick_random_images(self, count=4):
        # List all image files in the folder
        all_images = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # Check if enough images are available
        if len(all_images) < count:
            raise FileNotFoundError(f"Not enough images! Needed {count}, but found {len(all_images)}.")

        # Randomly pick 4 images (one for each direction)
        return random.sample(all_images, count)

    def calculate_vehicle_counts_with_images(self):
        # Get 4 random images
        images = self.pick_random_images()

        counts = {}           # To store vehicle counts for each direction
        annotated_images = []  # To store images with boxes drawn

        # Process each image
        for i, image_name in enumerate(images):
            image_path = os.path.join(self.image_folder, image_name)

            # Detect vehicles and get count + annotated image
            count, annotated_image, _ = self.detector.detect_and_count_with_image(image_path)

            # Store count as "Direction_1", "Direction_2", etc.
            counts[f"Direction_{i+1}"] = count
            annotated_images.append(annotated_image)

        return counts, annotated_images

    def decide_signal_timing(self, counts):
        # Minimum green light time for all signals
        minimum_time = 10

        # Find the direction with the most vehicles
        max_count = max(counts.values(), default=0)

        # Set time for each signal based on how much traffic it has
        timings = {}
        for direction, count in counts.items():
            if max_count > 0:
                extra_time = int((count / max_count) * 20)  # Scale time based on traffic
            else:
                extra_time = 0
            timings[direction] = minimum_time + extra_time

        return timings

    def run_control_cycle(self):
        # Step 1: Detect vehicles and get counts + images
        counts, annotated_images = self.calculate_vehicle_counts_with_images()

        # Step 2: Decide signal timing based on vehicle counts
        timings = self.decide_signal_timing(counts)

        # Return counts, timings, and images (for display if needed)
        return counts, timings, annotated_images
