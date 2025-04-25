import cv2
import os
from pathlib import Path

class VideoFrameExtractor:
    def __init__(self, video_dir='../data/videos', output_dir='../data/images'):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(self, video_name, frame_interval=30):
        """
        Extract frames from a video file
        
        Args:
            video_name (str): Name of the video file
            frame_interval (int): Extract one frame every N frames
        
        Returns:
            list: List of paths to extracted frames
        """
        video_path = self.video_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception(f"Error opening video file: {video_path}")

        frame_count = 0
        saved_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Generate frame filename
                frame_name = f"{video_name.stem}_frame_{frame_count}.jpg"
                frame_path = self.output_dir / frame_name
                
                # Save frame
                cv2.imwrite(str(frame_path), frame)
                saved_frames.append(frame_path)

            frame_count += 1

        cap.release()
        return saved_frames

    def process_all_videos(self, frame_interval=30):
        """
        Process all videos in the video directory
        
        Args:
            frame_interval (int): Extract one frame every N frames
        
        Returns:
            dict: Dictionary mapping video names to lists of extracted frame paths
        """
        results = {}
        
        # Process each video file in the directory
        for video_file in self.video_dir.glob('*.mp4'):
            try:
                frames = self.extract_frames(video_file.name, frame_interval)
                results[video_file.name] = frames
            except Exception as e:
                print(f"Error processing {video_file.name}: {str(e)}")
        
        return results

def main():
    # Example usage
    extractor = VideoFrameExtractor()
    
    # Process all videos
    results = extractor.process_all_videos(frame_interval=30)
    
    # Print results
    for video_name, frames in results.items():
        print(f"\nProcessed {video_name}:")
        print(f"Extracted {len(frames)} frames")

if __name__ == "__main__":
    main()