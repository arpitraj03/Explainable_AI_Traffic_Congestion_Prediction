"""
preprocessing.py
================
CCTV Video Preprocessing Pipeline

Responsibilities:
  - Open video files with OpenCV
  - Extract frames at configurable intervals
  - Apply background subtraction (MOG2) to isolate moving vehicles
  - Resize and normalise frames
  - Package frames into fixed-length sequences for the CNN-LSTM model
  - Save processed sequences to disk as NumPy arrays
"""

import os
import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional

# ── Logging setup ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load project configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════
#  BackgroundSubtractor
# ══════════════════════════════════════════════════════════════
class BackgroundSubtractor:
    """
    Wraps OpenCV's MOG2 background subtraction.

    MOG2 (Mixture of Gaussians v2) models the background as a mixture
    of Gaussians and updates it over time.  The foreground mask isolates
    moving vehicles, reducing the influence of static road features on
    the CNN.

    Args:
        history   : Number of frames used to model the background.
        var_threshold: Mahalanobis distance threshold for foreground/background.
        detect_shadows: Whether to mark shadows as a separate foreground class.
    """

    def __init__(self, history: int = 200, var_threshold: float = 50.0,
                 detect_shadows: bool = False):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply background subtraction to a single frame.

        Returns a 3-channel frame where static background pixels are
        darkened and moving (vehicle) pixels retain their colour.
        """
        fg_mask = self.subtractor.apply(frame)                  # binary mask
        fg_mask_3ch = cv2.merge([fg_mask, fg_mask, fg_mask])    # expand to 3ch
        result = cv2.bitwise_and(frame, fg_mask_3ch)            # mask frame
        return result


# ══════════════════════════════════════════════════════════════
#  FrameExtractor
# ══════════════════════════════════════════════════════════════
class FrameExtractor:
    """
    Extracts, preprocesses, and sequences frames from a video file.

    Processing steps per frame:
      1. BGR → RGB conversion
      2. (Optional) Background subtraction
      3. Resize to (height, width)
      4. Normalise pixel values to [0, 1]

    Frames are then grouped into non-overlapping sequences of
    `sequence_length` frames each.

    Args:
        height          : Target frame height (pixels).
        width           : Target frame width (pixels).
        sequence_length : Number of frames per sample fed to the model.
        frame_skip      : Sample every N-th frame to reduce temporal redundancy.
        use_bg_sub      : Apply MOG2 background subtraction.
        normalize       : Divide by 255 to normalise to [0, 1].
    """

    def __init__(self, height: int = 224, width: int = 224,
                 sequence_length: int = 16, frame_skip: int = 2,
                 use_bg_sub: bool = True, normalize: bool = True):
        self.height = height
        self.width = width
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.normalize = normalize
        self.bg_sub = BackgroundSubtractor() if use_bg_sub else None

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply all per-frame preprocessing steps."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # BGR → RGB
        if self.bg_sub is not None:
            frame = self.bg_sub.apply(frame)                    # BG subtraction
        frame = cv2.resize(frame, (self.width, self.height),    # resize
                           interpolation=cv2.INTER_AREA)
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0            # normalise
        return frame

    def extract_sequences(self, video_path: str) -> List[np.ndarray]:
        """
        Extract fixed-length frame sequences from a video.

        Returns:
            List of arrays, each with shape
            (sequence_length, height, width, 3).
            Returns an empty list if the video cannot be read.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return []

        frames: List[np.ndarray] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Only process every `frame_skip`-th frame
            if frame_idx % self.frame_skip == 0:
                frames.append(self._preprocess_frame(frame))
            frame_idx += 1

        cap.release()

        # Group frames into sequences
        sequences = []
        for start in range(0, len(frames) - self.sequence_length + 1,
                           self.sequence_length):
            seq = np.stack(frames[start: start + self.sequence_length], axis=0)
            sequences.append(seq)  # shape: (T, H, W, 3)

        logger.info(f"  {Path(video_path).name}: {len(frames)} frames "
                    f"→ {len(sequences)} sequences")
        return sequences

    def get_video_fps(self, video_path: str) -> float:
        """Return the FPS of the video (useful for real-time inference)."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    def get_video_metadata(self, video_path: str) -> dict:
        """Return basic metadata for a video file."""
        cap = cv2.VideoCapture(video_path)
        meta = {
            "fps":    cap.get(cv2.CAP_PROP_FPS),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_s": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) /
                          max(cap.get(cv2.CAP_PROP_FPS), 1),
        }
        cap.release()
        return meta


# ══════════════════════════════════════════════════════════════
#  VideoDatasetBuilder
# ══════════════════════════════════════════════════════════════
class VideoDatasetBuilder:
    """
    Scans a directory of labelled CCTV videos, extracts sequences,
    and saves (X, y) NumPy arrays to disk.

    Label detection:
        The class label is inferred from the video filename.
        Expected format: `<any_prefix>_<label>_<id>.mp4`
        Example:  `highway_high_003.mp4`  → label = "high"

    Args:
        raw_dir      : Path to directory containing raw .mp4 / .avi files.
        processed_dir: Destination directory for .npy output files.
        config       : Parsed config dict.
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(self, raw_dir: str, processed_dir: str, config: dict):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config
        self.class_to_idx = config["dataset"]["class_to_idx"]
        self.extractor = FrameExtractor(
            height=config["preprocessing"]["frame_height"],
            width=config["preprocessing"]["frame_width"],
            sequence_length=config["preprocessing"]["sequence_length"],
            frame_skip=config["preprocessing"]["frame_skip"],
            use_bg_sub=config["preprocessing"]["background_subtraction"],
            normalize=config["preprocessing"]["normalize"],
        )

    def _infer_label(self, filename: str) -> Optional[str]:
        """
        Infer congestion label from filename.

        Looks for any of the class names as a substring in the filename.
        Returns None if no label is found (file will be skipped).
        """
        name_lower = filename.lower()
        for label in self.class_to_idx:
            if label in name_lower:
                return label
        return None

    def build(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process all videos in raw_dir and return (X, y).

        Returns:
            X : ndarray of shape (N, T, H, W, 3)
            y : ndarray of shape (N,) with integer class indices
        """
        video_files = [
            f for f in self.raw_dir.iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not video_files:
            logger.warning(
                f"No supported video files found in {self.raw_dir}.\n"
                "Please add .mp4 / .avi files with labels in their names "
                "(e.g. traffic_high_001.mp4)."
            )
            # Return synthetic demo data so the pipeline can be tested
            return self._generate_demo_data()

        all_sequences: List[np.ndarray] = []
        all_labels: List[int] = []

        for vf in tqdm(video_files, desc="Processing videos"):
            label = self._infer_label(vf.stem)
            if label is None:
                logger.warning(f"Skipping {vf.name}: no label found in name.")
                continue

            sequences = self.extractor.extract_sequences(str(vf))
            label_idx = self.class_to_idx[label]

            all_sequences.extend(sequences)
            all_labels.extend([label_idx] * len(sequences))

        if not all_sequences:
            logger.warning("No sequences extracted. Generating demo data.")
            return self._generate_demo_data()

        X = np.stack(all_sequences, axis=0)   # (N, T, H, W, 3)
        y = np.array(all_labels, dtype=np.int32)

        # Save to disk
        np.save(self.processed_dir / "X.npy", X)
        np.save(self.processed_dir / "y.npy", y)
        logger.info(f"Saved dataset: X{X.shape}, y{y.shape} → {self.processed_dir}")

        return X, y

    def _generate_demo_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic frame sequences for pipeline testing.

        Creates dummy video sequences where:
          - Low congestion  → dark frames (sparse vehicles)
          - Medium          → medium brightness
          - High            → bright / dense frames
        """
        logger.info("Generating synthetic demo dataset (no real videos found).")
        cfg = self.cfg["preprocessing"]
        T = cfg["sequence_length"]
        H = cfg["frame_height"]
        W = cfg["frame_width"]

        rng = np.random.default_rng(42)
        samples_per_class = 20
        sequences, labels = [], []

        for cls_idx, (label, brightness) in enumerate(
                [("low", 0.2), ("medium", 0.5), ("high", 0.8)]):
            for _ in range(samples_per_class):
                # Simulate vehicle density via pixel brightness + structured noise
                base = rng.normal(brightness, 0.05, (T, H, W, 3)).clip(0, 1)
                # Add rectangular "vehicle blobs"
                for t in range(T):
                    num_vehicles = int(cls_idx * 8 + rng.integers(0, 6))
                    for _ in range(num_vehicles):
                        x1 = rng.integers(0, W - 30)
                        y1 = rng.integers(0, H - 15)
                        base[t, y1:y1+15, x1:x1+30] = rng.uniform(0.6, 1.0, (15, 30, 3))
                sequences.append(base.astype(np.float32))
                labels.append(cls_idx)

        X = np.stack(sequences, axis=0)
        y = np.array(labels, dtype=np.int32)

        np.save(self.processed_dir / "X.npy", X)
        np.save(self.processed_dir / "y.npy", y)
        logger.info(f"Demo dataset: X{X.shape}, y{y.shape}")
        return X, y


# ── CLI entry point ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess CCTV videos")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video_dir", default=None,
                        help="Override raw video directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = args.video_dir or cfg["paths"]["raw_video_dir"]
    proc_dir = cfg["paths"]["processed_dir"]

    builder = VideoDatasetBuilder(raw_dir, proc_dir, cfg)
    X, y = builder.build()

    class_counts = {k: int((y == v).sum())
                    for k, v in cfg["dataset"]["class_to_idx"].items()}
    logger.info(f"Class distribution: {class_counts}")
