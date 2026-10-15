"""
dataset.py
==========
Dataset Loading, Splitting, and TensorFlow Data Pipeline

Loads preprocessed (X, y) NumPy arrays, performs stratified
train / val / test splitting, applies one-hot encoding, and
wraps everything in tf.data.Dataset objects with online
augmentation support.
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
    Performs stratified train / val / test splitting.

    Stratification ensures that each split preserves the class
    distribution of the full dataset (important for imbalanced
    congestion levels).

    Args:
        train_ratio : Fraction of data for training.
        val_ratio   : Fraction of data for validation.
        test_ratio  : Fraction of data for testing (implicit).
        seed        : Random seed for reproducibility.
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
        """
        Split into (X_train, y_train), (X_val, y_val), (X_test, y_test).
        """
        # First split: train vs (val + test)
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y,
            test_size=(1.0 - self.train_ratio),
            stratify=y,
            random_state=self.seed
        )
        # Second split: val vs test
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
#  TrafficDataset  (tf.data pipeline)
# ══════════════════════════════════════════════════════════════
class TrafficDataset:
    """
    Wraps (X, y) arrays in a tf.data.Dataset with:
      - One-hot encoding of labels
      - Optional online augmentation (train set only)
      - Batching, shuffling, prefetching

    Args:
        X           : Float32 array (N, T, H, W, 3).
        y           : Int32 array (N,).
        num_classes : Number of congestion classes.
        batch_size  : Batch size.
        shuffle     : Shuffle the dataset each epoch.
        augmentor   : Optional SequenceAugmentor for online augmentation.
        repeat      : Repeat the dataset indefinitely (for training).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 num_classes: int = 3,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 augmentor: Optional[SequenceAugmentor] = None,
                 repeat: bool = True):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int32)
        self.num_classes = num_classes
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.augmentor   = augmentor
        self.repeat      = repeat

    def _aug_fn(self, x: tf.Tensor, y: tf.Tensor):
        """Apply augmentation inside the tf.data pipeline via py_function."""
        def _py_aug(x_np):
            rng = np.random.default_rng()
            return self.augmentor.augment(x_np.numpy(), rng)

        if self.augmentor is not None:
            x = tf.py_function(_py_aug, [x], tf.float32)
            x.set_shape(self.X.shape[1:])
        return x, y

    def build(self) -> tf.data.Dataset:
        """Construct and return the tf.data.Dataset."""
        y_onehot = tf.keras.utils.to_categorical(self.y, self.num_classes)
        ds = tf.data.Dataset.from_tensor_slices(
            (self.X, y_onehot.astype(np.float32))
        )

        if self.shuffle:
            ds = ds.shuffle(buffer_size=len(self.X), reshuffle_each_iteration=True)

        if self.augmentor is not None:
            ds = ds.map(self._aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        if self.repeat:
            ds = ds.repeat()

        return ds

    @property
    def steps_per_epoch(self) -> int:
        return max(1, len(self.X) // self.batch_size)


# ══════════════════════════════════════════════════════════════
#  DataManager  (convenience wrapper)
# ══════════════════════════════════════════════════════════════
class DataManager:
    """
    High-level convenience class that:
      1. Loads processed (or augmented) .npy arrays
      2. Splits them
      3. Builds and returns tf.data pipelines

    Args:
        cfg : Full config dict.
        use_augmented : Load augmented_X.npy instead of X.npy.
    """

    def __init__(self, cfg: dict, use_augmented: bool = False):
        self.cfg = cfg
        self.use_augmented = use_augmented
        self.num_classes = cfg["model"]["num_classes"]

    def load_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load X and y from disk."""
        if self.use_augmented:
            base = Path(self.cfg["paths"]["augmented_dir"])
            X = np.load(base / "augmented_X.npy")
            y = np.load(base / "augmented_y.npy")
        else:
            base = Path(self.cfg["paths"]["processed_dir"])
            X = np.load(base / "X.npy")
            y = np.load(base / "y.npy")

        logger.info(f"Loaded: X{X.shape}, y{y.shape} from {base}")
        return X, y

    def get_class_weights(self, y_train: np.ndarray) -> dict:
        """Compute inverse-frequency class weights for imbalanced datasets."""
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        return {int(c): float(w) for c, w in zip(classes, weights)}

    def prepare(self) -> dict:
        """
        Full preparation pipeline.

        Returns a dict with keys:
          train_ds, val_ds, test_ds,
          X_test, y_test,
          class_weights, steps_per_epoch
        """
        X, y = self.load_arrays()

        splitter = DataSplitter(
            train_ratio=self.cfg["dataset"]["train_ratio"],
            val_ratio=self.cfg["dataset"]["val_ratio"],
            test_ratio=self.cfg["dataset"]["test_ratio"],
            seed=self.cfg["dataset"]["random_seed"],
        )
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = splitter.split(X, y)

        aug_cfg = self.cfg.get("augmentation", {})
        aug = SequenceAugmentor(aug_cfg) if aug_cfg.get("enabled") else None

        bs = self.cfg["training"]["batch_size"]

        train_ds_obj = TrafficDataset(X_tr, y_tr, self.num_classes, bs,
                                      shuffle=True, augmentor=aug, repeat=True)
        val_ds_obj   = TrafficDataset(X_val, y_val, self.num_classes, bs,
                                      shuffle=False, augmentor=None, repeat=False)
        test_ds_obj  = TrafficDataset(X_te, y_te, self.num_classes, bs,
                                      shuffle=False, augmentor=None, repeat=False)

        cw = self.get_class_weights(y_tr) if self.cfg["training"]["class_weights"] else None

        return {
            "train_ds":        train_ds_obj.build(),
            "val_ds":          val_ds_obj.build(),
            "test_ds":         test_ds_obj.build(),
            "X_train": X_tr, "y_train": y_tr,
            "X_val":   X_val, "y_val":   y_val,
            "X_test":  X_te,  "y_test":  y_te,
            "class_weights":   cw,
            "steps_per_epoch": train_ds_obj.steps_per_epoch,
            "val_steps":       val_ds_obj.steps_per_epoch,
        }
