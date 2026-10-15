"""
dataset.py
==========
Dataset Loading, Splitting, and TensorFlow Data Pipeline

Fix v2:
    The original code used tf.data.Dataset.from_tensor_slices() which
    copies the ENTIRE array into TensorFlow's memory at once:
        437 val sequences × 502 MB = OOM crash

    This version uses tf.data.Dataset.from_generator() instead, which
    reads one sample at a time from the memory-mapped numpy array.
    Peak RAM during dataset construction is now a single batch (~few MB).
"""

import numpy as np
import yaml
import logging
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Optional

from augmentation import SequenceAugmentor

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  DataSplitter
# ══════════════════════════════════════════════════════════════
class DataSplitter:
    """
    Stratified train / val / test splitting.

    Stratification preserves the class distribution of the full
    dataset in each split — important for imbalanced congestion levels.
    """

    def __init__(self, train_ratio: float = 0.70,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.test_ratio  = test_ratio
        self.seed        = seed

    def split(self, X: np.ndarray, y: np.ndarray):
        """Return (X_train,y_train), (X_val,y_val), (X_test,y_test)."""
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y,
            test_size=(1.0 - self.train_ratio),
            stratify=y,
            random_state=self.seed
        )
        val_frac = self.val_ratio / (self.val_ratio + self.test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp,
            test_size=(1.0 - val_frac),
            stratify=y_tmp,
            random_state=self.seed
        )
        logger.info(
            f"Split → train:{len(X_train)} | val:{len(X_val)} | test:{len(X_test)}"
        )
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ══════════════════════════════════════════════════════════════
#  TrafficDataset  — generator-based tf.data pipeline
# ══════════════════════════════════════════════════════════════
class TrafficDataset:
    """
    Wraps (X, y) index arrays in a tf.data.Dataset using a Python
    generator so that data is read from the memory-mapped numpy file
    ONE SAMPLE AT A TIME rather than loading the whole split into RAM.

    This fixes the OOM that occurred when from_tensor_slices() tried
    to convert 437 sequences (502 MB) into a TensorFlow constant tensor.

    Args:
        X           : Float32 array or memmap (N, T, H, W, 3).
        y           : Int32 array (N,).
        num_classes : Number of congestion classes (for one-hot encoding).
        batch_size  : Training / inference batch size.
        shuffle     : Shuffle sample indices each epoch.
        augmentor   : Optional SequenceAugmentor (applied per sample).
        repeat      : Repeat indefinitely (True for train, False for val/test).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 num_classes: int = 3,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 augmentor: Optional[SequenceAugmentor] = None,
                 repeat: bool = True):
        # Store as numpy arrays (memmap-compatible — no full copy into RAM)
        self.X           = X
        self.y           = y.astype(np.int32)
        self.num_classes = num_classes
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.augmentor   = augmentor
        self.repeat      = repeat
        self.n           = len(X)

        # Pre-compute one-hot labels (small: N × 3 floats)
        self.y_onehot = np.eye(num_classes, dtype=np.float32)[self.y]

        # Infer shapes from first sample
        self._seq_shape = X[0].shape   # (T, H, W, 3)

    def _make_generator(self):
        """
        Returns a generator function that yields (sequence, label) pairs.

        The generator shuffles the index list each time it is called
        (when repeat=True and shuffle=True), ensuring different ordering
        each epoch without loading all data into memory.
        """
        indices = np.arange(self.n)

        def generator():
            idx = indices.copy()
            if self.shuffle:
                np.random.shuffle(idx)
            for i in idx:
                # Read one sequence from memmap (page-fault loads just this slice)
                x = np.array(self.X[i], dtype=np.float32)  # (T, H, W, 3)

                # Optional online augmentation
                if self.augmentor is not None:
                    rng = np.random.default_rng()
                    x = self.augmentor.augment(x, rng)

                y = self.y_onehot[i]   # (num_classes,)
                yield x, y

        return generator

    def build(self) -> tf.data.Dataset:
        """
        Construct and return the tf.data.Dataset.

        Uses from_generator() so TensorFlow never needs to hold the
        entire split in memory — samples are streamed lazily.
        """
        T, H, W, C = self._seq_shape

        ds = tf.data.Dataset.from_generator(
            self._make_generator(),
            output_signature=(
                tf.TensorSpec(shape=(T, H, W, C), dtype=tf.float32),
                tf.TensorSpec(shape=(self.num_classes,),  dtype=tf.float32),
            )
        )

        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        if self.repeat:
            ds = ds.repeat()

        return ds

    @property
    def steps_per_epoch(self) -> int:
        return max(1, self.n // self.batch_size)


# ══════════════════════════════════════════════════════════════
#  DataManager
# ══════════════════════════════════════════════════════════════
class DataManager:
    """
    High-level wrapper that:
      1. Loads processed (or augmented) .npy arrays as memmaps
      2. Splits them into train / val / test index subsets
      3. Builds generator-based tf.data pipelines for each split

    Args:
        cfg           : Full project config dict.
        use_augmented : If True, load from augmented_dir instead of processed_dir.
    """

    def __init__(self, cfg: dict, use_augmented: bool = False):
        self.cfg           = cfg
        self.use_augmented = use_augmented
        self.num_classes   = cfg["model"]["num_classes"]

    def load_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load X and y from disk as memory-mapped arrays.

        mmap_mode="r" means numpy reads pages from disk on demand —
        the full array is NEVER copied into RAM.
        """
        if self.use_augmented:
            base = Path(self.cfg["paths"]["augmented_dir"])
            X = np.load(base / "augmented_X.npy", mmap_mode="r")
            y = np.load(base / "augmented_y.npy")
        else:
            base = Path(self.cfg["paths"]["processed_dir"])
            X = np.load(base / "X.npy", mmap_mode="r")
            y = np.load(base / "y.npy")

        logger.info(f"Loaded: X{X.shape}, y{y.shape} from {base}")
        return X, y

    def get_class_weights(self, y_train: np.ndarray) -> dict:
        """Compute inverse-frequency class weights for imbalanced data."""
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        return {int(c): float(w) for c, w in zip(classes, weights)}

    def prepare(self) -> dict:
        """
        Full pipeline: load → split → build datasets.

        Returns a dict with:
            train_ds, val_ds, test_ds   : tf.data.Dataset objects
            X_train/val/test            : numpy index arrays (memmap)
            y_train/val/test            : numpy label arrays
            class_weights               : dict or None
            steps_per_epoch, val_steps  : int
        """
        X, y = self.load_arrays()

        splitter = DataSplitter(
            train_ratio = self.cfg["dataset"]["train_ratio"],
            val_ratio   = self.cfg["dataset"]["val_ratio"],
            test_ratio  = self.cfg["dataset"]["test_ratio"],
            seed        = self.cfg["dataset"]["random_seed"],
        )
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = splitter.split(X, y)

        aug_cfg = self.cfg.get("augmentation", {})
        aug     = SequenceAugmentor(aug_cfg) if aug_cfg.get("enabled") else None
        bs      = self.cfg["training"]["batch_size"]

        train_ds_obj = TrafficDataset(X_tr,  y_tr,  self.num_classes, bs,
                                      shuffle=True,  augmentor=aug, repeat=True)
        val_ds_obj   = TrafficDataset(X_val, y_val, self.num_classes, bs,
                                      shuffle=False, augmentor=None, repeat=False)
        test_ds_obj  = TrafficDataset(X_te,  y_te,  self.num_classes, bs,
                                      shuffle=False, augmentor=None, repeat=False)

        cw = (self.get_class_weights(y_tr)
              if self.cfg["training"]["class_weights"] else None)

        return {
            "train_ds":        train_ds_obj.build(),
            "val_ds":          val_ds_obj.build(),
            "test_ds":         test_ds_obj.build(),
            "X_train": X_tr,  "y_train": y_tr,
            "X_val":   X_val, "y_val":   y_val,
            "X_test":  X_te,  "y_test":  y_te,
            "class_weights":   cw,
            "steps_per_epoch": train_ds_obj.steps_per_epoch,
            "val_steps":       val_ds_obj.steps_per_epoch,
        }
