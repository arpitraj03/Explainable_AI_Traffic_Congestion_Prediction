"""
augmentation.py
===============
Data Augmentation for Video Frame Sequences

Applies consistent augmentations across all frames in a sequence so that
temporal coherence is preserved (the same random flip/brightness is applied
to every frame in the clip, not independently per frame).

Augmentation strategies implemented:
  - Horizontal flip
  - Random brightness & contrast adjustment
  - Small rotation
  - Zoom / crop
  - Gaussian noise injection
"""

import numpy as np
import cv2
import yaml
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  SequenceAugmentor
# ══════════════════════════════════════════════════════════════
class SequenceAugmentor:
    """
    Applies temporally-consistent augmentation to a (T, H, W, 3) sequence.

    All random parameters are sampled ONCE per sequence, then applied
    identically to every frame.  This is critical for video data:
    if frames in the same clip had independent random flips, the model
    would see physically impossible transitions.

    Args:
        cfg : The 'augmentation' sub-dict from config.yaml.
    """

    def __init__(self, cfg: dict):
        self.enabled          = cfg.get("enabled", True)
        self.h_flip           = cfg.get("horizontal_flip", True)
        self.brightness_range = cfg.get("brightness_range", [0.8, 1.2])
        self.contrast_range   = cfg.get("contrast_range",   [0.8, 1.2])
        self.rotation_deg     = cfg.get("rotation_degrees", 10)
        self.zoom_range       = cfg.get("zoom_range", 0.1)
        self.noise_std        = cfg.get("noise_std", 0.02)

    def augment(self, sequence: np.ndarray,
                rng: np.random.Generator = None) -> np.ndarray:
        """
        Augment a single sequence.

        Args:
            sequence : float32 array of shape (T, H, W, 3) in [0, 1].
            rng      : Optional seeded RNG for reproducibility.

        Returns:
            Augmented sequence with the same shape.
        """
        if not self.enabled:
            return sequence

        if rng is None:
            rng = np.random.default_rng()

        T, H, W, C = sequence.shape
        aug = sequence.copy()

        # ── 1. Horizontal flip (50 % probability) ─────────────
        if self.h_flip and rng.random() < 0.5:
            aug = aug[:, :, ::-1, :]   # flip width axis for all T frames

        # ── 2. Brightness adjustment ───────────────────────────
        brightness = rng.uniform(*self.brightness_range)
        aug = (aug * brightness).clip(0.0, 1.0)

        # ── 3. Contrast adjustment ─────────────────────────────
        contrast = rng.uniform(*self.contrast_range)
        mean = aug.mean(axis=(1, 2, 3), keepdims=True)
        aug = ((aug - mean) * contrast + mean).clip(0.0, 1.0)

        # ── 4. Rotation ────────────────────────────────────────
        if self.rotation_deg > 0 and rng.random() < 0.5:
            angle = rng.uniform(-self.rotation_deg, self.rotation_deg)
            center = (W / 2, H / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug = np.stack([
                cv2.warpAffine(aug[t], M, (W, H),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
                for t in range(T)
            ], axis=0)

        # ── 5. Zoom / random crop ──────────────────────────────
        if self.zoom_range > 0 and rng.random() < 0.5:
            zoom = rng.uniform(0.0, self.zoom_range)
            crop_h = int(H * zoom / 2)
            crop_w = int(W * zoom / 2)
            if crop_h > 0 and crop_w > 0:
                aug = aug[:, crop_h:-crop_h, crop_w:-crop_w, :]
                aug = np.stack([
                    cv2.resize(aug[t], (W, H), interpolation=cv2.INTER_LINEAR)
                    for t in range(T)
                ], axis=0)

        # ── 6. Gaussian noise injection ────────────────────────
        if self.noise_std > 0:
            noise = rng.normal(0, self.noise_std, aug.shape).astype(np.float32)
            aug = (aug + noise).clip(0.0, 1.0)

        return aug.astype(np.float32)


# ══════════════════════════════════════════════════════════════
#  DatasetAugmentor (offline augmentation of saved .npy files)
# ══════════════════════════════════════════════════════════════
class DatasetAugmentor:
    """
    Loads a saved (X, y) dataset, applies augmentation, and saves
    an expanded dataset.

    Offline augmentation is useful when the raw dataset is small.
    For large datasets, prefer online augmentation in the data loader.

    Args:
        processed_dir : Directory containing X.npy and y.npy.
        augmented_dir : Destination for augmented_X.npy / augmented_y.npy.
        cfg           : Full config dict.
        copies        : Number of augmented copies to generate per sample.
    """

    def __init__(self, processed_dir: str, augmented_dir: str,
                 cfg: dict, copies: int = 2):
        self.processed_dir = Path(processed_dir)
        self.augmented_dir = Path(augmented_dir)
        self.augmented_dir.mkdir(parents=True, exist_ok=True)
        self.augmentor = SequenceAugmentor(cfg["augmentation"])
        self.copies = copies

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load X.npy and y.npy, augment, concatenate with originals,
        and save augmented_X.npy / augmented_y.npy.
        """
        X = np.load(self.processed_dir / "X.npy")
        y = np.load(self.processed_dir / "y.npy")

        logger.info(f"Original dataset: X{X.shape}, y{y.shape}")
        rng = np.random.default_rng(42)

        aug_X_list = [X]
        aug_y_list = [y]

        for copy_idx in range(self.copies):
            logger.info(f"Generating augmentation copy {copy_idx + 1}/{self.copies}")
            aug_batch = np.stack([
                self.augmentor.augment(X[i], rng) for i in range(len(X))
            ], axis=0)
            aug_X_list.append(aug_batch)
            aug_y_list.append(y)

        aug_X = np.concatenate(aug_X_list, axis=0)
        aug_y = np.concatenate(aug_y_list, axis=0)

        # Shuffle
        perm = rng.permutation(len(aug_X))
        aug_X, aug_y = aug_X[perm], aug_y[perm]

        np.save(self.augmented_dir / "augmented_X.npy", aug_X)
        np.save(self.augmented_dir / "augmented_y.npy", aug_y)
        logger.info(f"Saved augmented dataset: X{aug_X.shape}, y{aug_y.shape}")

        return aug_X, aug_y


# ── CLI entry point ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augment training dataset")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--copies", type=int, default=2,
                        help="Number of augmented copies per sample")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    augmentor = DatasetAugmentor(
        processed_dir=cfg["paths"]["processed_dir"],
        augmented_dir=cfg["paths"]["augmented_dir"],
        cfg=cfg,
        copies=args.copies,
    )
    augmentor.run()
