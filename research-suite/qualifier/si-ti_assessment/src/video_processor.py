"""Video processing module for extracting frames from video files."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Generator
from tqdm import tqdm


class VideoProcessor:
    """Handles video file reading and frame extraction."""

    SUPPORTED_FORMATS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')

    def __init__(self, video_path: str):
        """
        Initialize video processor.

        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_video_info(self) -> dict:
        """
        Get video metadata.

        Returns:
            Dictionary containing video information
        """
        return {
            'filename': self.video_path.name,
            'path': str(self.video_path),
            'fps': self.fps,
            'total_frames': self.total_frames,
            'width': self.width,
            'height': self.height,
            'duration_sec': self.total_frames / self.fps if self.fps > 0 else 0
        }

    def extract_frames(self, show_progress: bool = True) -> Generator[np.ndarray, None, None]:
        """
        Extract all frames from video.

        Args:
            show_progress: Whether to show progress bar

        Yields:
            Video frames as numpy arrays (BGR format)
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

        iterator = range(self.total_frames)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Processing {self.video_path.name}", unit="frames")

        for _ in iterator:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


def find_video_files(directory: str) -> List[Path]:
    """
    Find all video files in a directory.

    Args:
        directory: Path to directory containing videos

    Returns:
        List of paths to video files
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    video_files = []
    for ext in VideoProcessor.SUPPORTED_FORMATS:
        video_files.extend(dir_path.glob(f"*{ext}"))
        video_files.extend(dir_path.glob(f"*{ext.upper()}"))

    return sorted(video_files)
