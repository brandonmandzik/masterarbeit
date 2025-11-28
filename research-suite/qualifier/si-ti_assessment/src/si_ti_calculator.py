"""SI/TI calculation module following ITU-T Recommendation P.910."""

import numpy as np
from scipy import ndimage
from typing import List, Tuple, Dict


class SITICalculator:
    """
    Calculate Spatial Information (SI) and Temporal Information (TI)
    according to ITU-T Recommendation P.910.
    """

    @staticmethod
    def _convert_to_grayscale(frame: np.ndarray) -> np.ndarray:
        """
        Convert BGR frame to grayscale using ITU-R BT.601 luma formula.

        Formula: Y = 0.299*R + 0.587*G + 0.114*B

        Args:
            frame: BGR frame (OpenCV format)

        Returns:
            Grayscale frame as float64
        """
        if len(frame.shape) == 2:
            # Already grayscale
            return frame.astype(np.float64)

        # OpenCV uses BGR, so indices are: B=0, G=1, R=2
        # ITU-R BT.601: Y = 0.299*R + 0.587*G + 0.114*B
        gray = (0.299 * frame[:, :, 2] +
                0.587 * frame[:, :, 1] +
                0.114 * frame[:, :, 0])

        return gray.astype(np.float64)

    @staticmethod
    def calculate_si(frame: np.ndarray) -> float:
        """
        Calculate Spatial Information (SI) for a single frame.

        SI measures the spatial complexity using Sobel edge detection.

        Formula:
            SI = stddev(Sobel(Frame))

        Where:
            - Sobel() applies horizontal and vertical edge filters
            - Combined as: sqrt(Gx^2 + Gy^2)
            - stddev() is standard deviation across all pixels

        Args:
            frame: Video frame (BGR or grayscale)

        Returns:
            SI value (spatial information)
        """
        # Convert to grayscale
        gray = SITICalculator._convert_to_grayscale(frame)

        # Apply Sobel filter in both directions
        # scipy.ndimage.sobel uses the following kernels:
        # Horizontal (axis=0): [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        # Vertical (axis=1): [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_x = ndimage.sobel(gray, axis=1)  # Vertical edges
        sobel_y = ndimage.sobel(gray, axis=0)  # Horizontal edges

        # Combine gradients: magnitude = sqrt(Gx^2 + Gy^2)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Calculate standard deviation (spatial information)
        si = np.std(sobel_magnitude)

        return float(si)

    @staticmethod
    def calculate_ti(frame_current: np.ndarray, frame_previous: np.ndarray) -> float:
        """
        Calculate Temporal Information (TI) between consecutive frames.

        TI measures temporal complexity based on frame differences.

        Formula:
            TI = stddev(Frame_n - Frame_{n-1})

        Args:
            frame_current: Current frame
            frame_previous: Previous frame

        Returns:
            TI value (temporal information)
        """
        # Convert both frames to grayscale
        gray_current = SITICalculator._convert_to_grayscale(frame_current)
        gray_previous = SITICalculator._convert_to_grayscale(frame_previous)

        # Calculate temporal difference
        frame_diff = gray_current - gray_previous

        # Calculate standard deviation (temporal information)
        ti = np.std(frame_diff)

        return float(ti)

    @staticmethod
    def process_video_frames(frames_generator) -> Dict[str, any]:
        """
        Process frames from a video and calculate SI/TI metrics.

        Args:
            frames_generator: Generator or iterable yielding video frames

        Returns:
            Dictionary containing:
                - si_values: List of SI values per frame
                - ti_values: List of TI values per frame pair
                - si_max: Maximum SI value
                - ti_max: Maximum TI value
                - si_mean: Mean SI value
                - ti_mean: Mean TI value
                - si_std: Standard deviation of SI values
                - ti_std: Standard deviation of TI values
        """
        si_values = []
        ti_values = []
        previous_frame = None

        # Process frames incrementally (memory efficient)
        for frame in frames_generator:
            # Calculate SI for current frame
            si = SITICalculator.calculate_si(frame)
            si_values.append(si)

            # Calculate TI if we have a previous frame
            if previous_frame is not None:
                ti = SITICalculator.calculate_ti(frame, previous_frame)
                ti_values.append(ti)

            # Store current frame as previous for next iteration
            previous_frame = frame

        if not si_values:
            raise ValueError("No frames provided")

        # Aggregate statistics
        results = {
            'si_values': si_values,
            'ti_values': ti_values,
            'si_max': float(np.max(si_values)),
            'ti_max': float(np.max(ti_values)) if ti_values else 0.0,
            'si_mean': float(np.mean(si_values)),
            'ti_mean': float(np.mean(ti_values)) if ti_values else 0.0,
            'si_std': float(np.std(si_values)),
            'ti_std': float(np.std(ti_values)) if ti_values else 0.0,
            'si_median': float(np.median(si_values)),
            'ti_median': float(np.median(ti_values)) if ti_values else 0.0,
        }

        return results
