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

"""
preprocessing.py  —  CCTV Video Preprocessing Pipeline
"""

import os
import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class BackgroundSubtractor:
    def __init__(self, history=200, var_threshold=50.0, detect_shadows=False):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold,
            detectShadows=detect_shadows
        )

    def apply(self, frame):
        fg_mask = self.subtractor.apply(frame)
        fg_mask_3ch = cv2.merge([fg_mask, fg_mask, fg_mask])
        return cv2.bitwise_and(frame, fg_mask_3ch)


class FrameExtractor:
    def __init__(self, height=112, width=112, sequence_length=8,
                 frame_skip=4, use_bg_sub=True, normalize=True):
        self.height = height
        self.width = width
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.normalize = normalize
        self.bg_sub = BackgroundSubtractor() if use_bg_sub else None

    def _preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.bg_sub is not None:
            frame = self.bg_sub.apply(frame)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        return frame

    def extract_sequences(self, video_path):
        video_path = str(Path(video_path).resolve())
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return []

        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.frame_skip == 0:
                frames.append(self._preprocess_frame(frame))
            frame_idx += 1
        cap.release()

        if not frames:
            logger.warning(f"No frames read from: {video_path}")
            return []

        sequences = []
        for start in range(0, len(frames) - self.sequence_length + 1,
                           self.sequence_length):
            seq = np.stack(frames[start: start + self.sequence_length], axis=0)
            sequences.append(seq)

        logger.info(f"  {Path(video_path).name}: {len(frames)} frames "
                    f"-> {len(sequences)} sequences")
        return sequences

    def get_video_fps(self, video_path):
        video_path = str(Path(video_path).resolve())
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return max(fps, 1.0)

    def get_video_metadata(self, video_path):
        video_path = str(Path(video_path).resolve())
        cap = cv2.VideoCapture(video_path)
        fps = max(cap.get(cv2.CAP_PROP_FPS), 1.0)
        meta = {
            "fps": fps,
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_s": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps,
        }
        cap.release()
        return meta


class VideoDatasetBuilder:
    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(self, raw_dir, processed_dir, config):
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

    def _infer_label(self, filename):
        name_lower = filename.lower()
        for label in self.class_to_idx:
            if label in name_lower:
                return label
        return None

    def build(self):
        video_files = [
            f for f in self.raw_dir.iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not video_files:
            logger.warning(f"No video files found in {self.raw_dir}.")
            return self._generate_demo_data()

        all_sequences = []
        all_labels = []

        for vf in tqdm(video_files, desc="Processing videos"):
            label = self._infer_label(vf.stem)
            if label is None:
                logger.warning(f"Skipping {vf.name}: no label in filename.")
                continue
            sequences = self.extractor.extract_sequences(str(vf))
            if not sequences:
                continue
            label_idx = self.class_to_idx[label]
            all_sequences.extend(sequences)
            all_labels.extend([label_idx] * len(sequences))

        if not all_sequences:
            logger.warning("No sequences extracted. Using demo data.")
            return self._generate_demo_data()

        chunk_size = 100
        n_total = len(all_sequences)
        seq_shape = all_sequences[0].shape
        X_path = self.processed_dir / "X.npy"
        y_path = self.processed_dir / "y.npy"

        logger.info(f"Writing {n_total} sequences in chunks of {chunk_size}...")

        X_mmap = np.lib.format.open_memmap(
            str(X_path), mode="w+", dtype=np.float32,
            shape=(n_total, *seq_shape)
        )

        for i in range(0, n_total, chunk_size):
            chunk_seqs = all_sequences[i: i + chunk_size]
            chunk_arr = np.stack(chunk_seqs, axis=0)
            end = i + len(chunk_seqs)
            X_mmap[i:end] = chunk_arr
            logger.info(f"  Saved {end} / {n_total} sequences")

        del X_mmap

        y = np.array(all_labels, dtype=np.int32)
        np.save(str(y_path), y)

        X = np.load(str(X_path), mmap_mode="r")
        logger.info(f"Dataset ready: X{X.shape}, y{y.shape}")

        for label, idx in self.class_to_idx.items():
            count = int((y == idx).sum())
            logger.info(f"  {label:>8}: {count} sequences")

        return X, y

    def _generate_demo_data(self):
        logger.info("Generating synthetic demo dataset.")
        cfg = self.cfg["preprocessing"]
        T = cfg["sequence_length"]
        H = cfg["frame_height"]
        W = cfg["frame_width"]
        rng = np.random.default_rng(42)
        samples_per_class = 20
        sequences, labels = [], []

        for cls_idx, (_, brightness) in enumerate(
                [("low", 0.2), ("medium", 0.5), ("high", 0.8)]):
            for _ in range(samples_per_class):
                base = rng.normal(brightness, 0.05,
                                  (T, H, W, 3)).clip(0, 1).astype(np.float32)
                for t in range(T):
                    num_vehicles = int(cls_idx * 8 + rng.integers(0, 6))
                    for _ in range(num_vehicles):
                        x1 = int(rng.integers(0, max(W - 30, 1)))
                        y1 = int(rng.integers(0, max(H - 15, 1)))
                        x2 = min(x1 + 30, W)
                        y2 = min(y1 + 15, H)
                        base[t, y1:y2, x1:x2] = rng.uniform(
                            0.6, 1.0, (y2 - y1, x2 - x1, 3)
                        ).astype(np.float32)
                sequences.append(base)
                labels.append(cls_idx)

        X = np.stack(sequences, axis=0)
        y = np.array(labels, dtype=np.int32)
        np.save(self.processed_dir / "X.npy", X)
        np.save(self.processed_dir / "y.npy", y)
        logger.info(f"Demo dataset: X{X.shape}, y{y.shape}")
        return X, y


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video_dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw_dir = args.video_dir or cfg["paths"]["raw_video_dir"]
    builder = VideoDatasetBuilder(raw_dir, cfg["paths"]["processed_dir"], cfg)
    X, y = builder.build()
    logger.info({k: int((y == v).sum()) for k, v in cfg["dataset"]["class_to_idx"].items()})