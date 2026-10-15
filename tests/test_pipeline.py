"""
tests/test_pipeline.py
======================
Unit Tests for All Pipeline Modules

Run with: python -m pytest tests/ -v
"""

import sys
import os
import pytest
import numpy as np
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Minimal in-memory config for tests (no disk I/O required)
TEST_CFG = {
    "preprocessing": {
        "frame_height": 64,
        "frame_width":  64,
        "sequence_length": 4,
        "frame_skip": 1,
        "background_subtraction": False,
        "normalize": True,
        "num_channels": 3,
    },
    "dataset": {
        "train_ratio": 0.70,
        "val_ratio":   0.15,
        "test_ratio":  0.15,
        "random_seed": 42,
        "classes":     ["low", "medium", "high"],
        "class_to_idx": {"low": 0, "medium": 1, "high": 2},
    },
    "augmentation": {
        "enabled":          True,
        "horizontal_flip":  True,
        "brightness_range": [0.9, 1.1],
        "contrast_range":   [0.9, 1.1],
        "rotation_degrees": 5,
        "zoom_range":       0.05,
        "noise_std":        0.01,
    },
    "model": {
        "cnn_backbone":    "mobilenetv2",
        "cnn_trainable":   False,
        "cnn_feature_dim": 1280,
        "lstm_units":      [32, 16],
        "lstm_dropout":    0.0,
        "lstm_recurrent_dropout": 0.0,
        "dense_units":     [32],
        "dense_dropout":   0.0,
        "num_classes":     3,
        "activation":      "softmax",
    },
    "training": {
        "epochs":                    1,
        "batch_size":                4,
        "learning_rate":             0.001,
        "lr_scheduler":              "cosine",
        "optimizer":                 "adam",
        "loss":                      "categorical_crossentropy",
        "early_stopping_patience":   2,
        "reduce_lr_patience":        1,
        "reduce_lr_factor":          0.5,
        "class_weights":             True,
        "mixed_precision":           False,
    },
    "explainability": {
        "gradcam": {
            "enabled":      True,
            "target_layer": "block_16_project_BN",
            "colormap":     "jet",
            "alpha":        0.4,
        },
        "shap":  {"enabled": False, "num_background": 10, "num_explain": 2},
        "lime":  {"enabled": False, "num_samples": 50, "num_segments": 10, "num_features": 5},
    },
    "paths": {
        "raw_video_dir":    "data/raw",
        "processed_dir":    "/tmp/tc_test/processed",
        "augmented_dir":    "/tmp/tc_test/augmented",
        "model_save_dir":   "/tmp/tc_test/models",
        "output_vis_dir":   "/tmp/tc_test/vis",
        "heatmap_dir":      "/tmp/tc_test/heatmaps",
        "report_dir":       "/tmp/tc_test/reports",
        "model_checkpoint": "/tmp/tc_test/models/best.h5",
    },
}


# ══════════════════════════════════════════════════════════════
#  Preprocessing Tests
# ══════════════════════════════════════════════════════════════
class TestPreprocessing:

    def test_frame_extractor_returns_list(self, tmp_path):
        from preprocessing import FrameExtractor
        ext = FrameExtractor(
            height=64, width=64, sequence_length=4,
            frame_skip=1, use_bg_sub=False, normalize=True
        )
        # Non-existent video should return empty list
        result = ext.extract_sequences("/nonexistent/video.mp4")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_demo_data_generation(self, tmp_path):
        from preprocessing import VideoDatasetBuilder
        cfg = dict(TEST_CFG)
        cfg["paths"]["processed_dir"] = str(tmp_path)

        builder = VideoDatasetBuilder(
            raw_dir="/nonexistent", processed_dir=str(tmp_path), config=cfg
        )
        X, y = builder.build()

        assert X.ndim == 5, "X should be 5D: (N, T, H, W, C)"
        assert y.ndim == 1
        assert len(X) == len(y)
        assert X.shape[1] == cfg["preprocessing"]["sequence_length"]
        assert X.min() >= 0.0
        assert X.max() <= 1.0


# ══════════════════════════════════════════════════════════════
#  Augmentation Tests
# ══════════════════════════════════════════════════════════════
class TestAugmentation:

    def _make_sequence(self, T=4, H=64, W=64):
        rng = np.random.default_rng(0)
        return rng.uniform(0, 1, (T, H, W, 3)).astype(np.float32)

    def test_augmented_shape_preserved(self):
        from augmentation import SequenceAugmentor
        aug = SequenceAugmentor(TEST_CFG["augmentation"])
        seq = self._make_sequence()
        out = aug.augment(seq)
        assert out.shape == seq.shape

    def test_augmented_values_in_range(self):
        from augmentation import SequenceAugmentor
        aug = SequenceAugmentor(TEST_CFG["augmentation"])
        seq = self._make_sequence()
        out = aug.augment(seq)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_augmented_dtype_float32(self):
        from augmentation import SequenceAugmentor
        aug = SequenceAugmentor(TEST_CFG["augmentation"])
        seq = self._make_sequence()
        out = aug.augment(seq)
        assert out.dtype == np.float32

    def test_no_aug_returns_unchanged(self):
        from augmentation import SequenceAugmentor
        cfg = dict(TEST_CFG["augmentation"])
        cfg["enabled"] = False
        aug = SequenceAugmentor(cfg)
        seq = self._make_sequence()
        out = aug.augment(seq)
        np.testing.assert_array_equal(out, seq)


# ══════════════════════════════════════════════════════════════
#  Dataset Tests
# ══════════════════════════════════════════════════════════════
class TestDataset:

    def _make_Xy(self, N=60):
        rng = np.random.default_rng(7)
        X = rng.uniform(0, 1, (N, 4, 64, 64, 3)).astype(np.float32)
        y = rng.integers(0, 3, N).astype(np.int32)
        return X, y

    def test_split_sizes(self):
        from dataset import DataSplitter
        X, y = self._make_Xy(N=60)
        splitter = DataSplitter(0.70, 0.15, 0.15, seed=42)
        (Xtr, ytr), (Xv, yv), (Xte, yte) = splitter.split(X, y)
        total = len(Xtr) + len(Xv) + len(Xte)
        assert total == len(X), "All samples must be assigned to a split"

    def test_split_no_leakage(self):
        from dataset import DataSplitter
        X, y = self._make_Xy(N=60)
        splitter = DataSplitter(seed=42)
        (Xtr, _), (Xv, _), (Xte, _) = splitter.split(X, y)
        # No row should appear in two splits simultaneously
        tr_ids = set(map(lambda a: a.tobytes(), Xtr.reshape(len(Xtr), -1)))
        v_ids  = set(map(lambda a: a.tobytes(), Xv.reshape(len(Xv),  -1)))
        te_ids = set(map(lambda a: a.tobytes(), Xte.reshape(len(Xte), -1)))
        assert len(tr_ids & v_ids)  == 0, "Train/Val overlap detected"
        assert len(tr_ids & te_ids) == 0, "Train/Test overlap detected"
        assert len(v_ids  & te_ids) == 0, "Val/Test overlap detected"

    def test_tf_dataset_builds(self):
        import tensorflow as tf
        from dataset import TrafficDataset
        X, y = self._make_Xy(N=20)
        ds_obj = TrafficDataset(X, y, num_classes=3, batch_size=4, shuffle=False)
        ds = ds_obj.build()
        batch = next(iter(ds))
        Xb, yb = batch
        assert Xb.shape[0] <= 4
        assert yb.shape[-1] == 3   # one-hot


# ══════════════════════════════════════════════════════════════
#  Model Tests
# ══════════════════════════════════════════════════════════════
class TestModel:

    def test_model_builds(self):
        from model import CongestionModel
        builder = CongestionModel(TEST_CFG)
        model = builder.build()
        assert model is not None
        assert model.output_shape[-1] == 3

    def test_model_forward_pass(self):
        import tensorflow as tf
        from model import CongestionModel
        builder = CongestionModel(TEST_CFG)
        model   = builder.build()
        T, H, W = (TEST_CFG["preprocessing"]["sequence_length"],
                   TEST_CFG["preprocessing"]["frame_height"],
                   TEST_CFG["preprocessing"]["frame_width"])
        dummy_input = np.zeros((2, T, H, W, 3), dtype=np.float32)
        output = model.predict(dummy_input, verbose=0)
        assert output.shape == (2, 3)
        # Softmax probabilities must sum to 1
        np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-5)

    def test_model_output_range(self):
        from model import CongestionModel
        builder = CongestionModel(TEST_CFG)
        model   = builder.build()
        T, H, W = (4, 64, 64)
        dummy = np.random.rand(3, T, H, W, 3).astype(np.float32)
        out = model.predict(dummy, verbose=0)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ══════════════════════════════════════════════════════════════
#  Visualizer Tests
# ══════════════════════════════════════════════════════════════
class TestVisualizer:

    def test_frame_annotator(self):
        from visualizer import FrameAnnotator
        ann   = FrameAnnotator(frame_size=(64, 64))
        frame = np.random.rand(64, 64, 3).astype(np.float32)
        out   = ann.annotate(frame, pred_class=1, confidence=0.85)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8

    def test_timeline_plotter(self, tmp_path):
        from visualizer import TimelinePlotter
        preds = [0, 0, 1, 1, 2, 2, 1, 0]
        confs = [0.9, 0.8, 0.75, 0.85, 0.95, 0.92, 0.70, 0.88]
        TimelinePlotter.plot(
            preds, confs, "test_video.mp4",
            str(tmp_path), fps=30.0
        )
        out = tmp_path / "timeline_test_video.png"
        assert out.exists(), "Timeline image should be saved"


# ── Run standalone ─────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
