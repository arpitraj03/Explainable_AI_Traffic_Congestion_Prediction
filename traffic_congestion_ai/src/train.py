"""
train.py
========
Model Training Loop

Orchestrates:
  1. Data preparation (load, split, build tf.data pipelines)
  2. Model construction & compilation
  3. Optionally: offline augmentation
  4. Training with callbacks (early stopping, LR scheduling, checkpointing)
  5. Saving the final model
  6. Plotting training history
"""

import os
import sys
import yaml
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

# Add project root to path when running as script
sys.path.insert(0, str(Path(__file__).parent))

from dataset  import DataManager
from model    import CongestionModel, build_callbacks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  Trainer
# ══════════════════════════════════════════════════════════════
class Trainer:
    """
    End-to-end training orchestrator.

    Args:
        cfg          : Full config dict.
        use_augmented: If True, load augmented dataset instead of raw processed.
    """

    def __init__(self, cfg: dict, use_augmented: bool = False):
        self.cfg          = cfg
        self.use_augmented = use_augmented
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Create output directories if they don't exist."""
        for key in ["model_save_dir", "report_dir", "output_vis_dir", "heatmap_dir"]:
            Path(self.cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    def run(self) -> tf.keras.Model:
        """
        Full training pipeline.

        Returns:
            Trained Keras model.
        """
        logger.info("=" * 60)
        logger.info("  TRAFFIC CONGESTION AI  —  Training Pipeline")
        logger.info("=" * 60)

        # ── 1. Prepare data ────────────────────────────────────
        logger.info("[1/5] Preparing data...")
        dm = DataManager(self.cfg, use_augmented=self.use_augmented)
        data = dm.prepare()

        logger.info(
            f"  Train  : {len(data['X_train'])} sequences\n"
            f"  Val    : {len(data['X_val'])} sequences\n"
            f"  Test   : {len(data['X_test'])} sequences\n"
            f"  Class weights: {data['class_weights']}"
        )

        # ── 2. Build model ─────────────────────────────────────
        logger.info("[2/5] Building model...")
        builder = CongestionModel(self.cfg)
        model   = builder.build()

        # ── 3. Build callbacks ─────────────────────────────────
        logger.info("[3/5] Configuring callbacks...")
        callbacks = build_callbacks(self.cfg)

        # ── 4. Train ───────────────────────────────────────────
        tc = self.cfg["training"]
        logger.info(f"[4/5] Training for up to {tc['epochs']} epochs "
                    f"(batch_size={tc['batch_size']})...")

        history = model.fit(
            data["train_ds"],
            epochs=tc["epochs"],
            steps_per_epoch=data["steps_per_epoch"],
            validation_data=data["val_ds"],
            validation_steps=data["val_steps"],
            class_weight=data["class_weights"],
            callbacks=callbacks,
            verbose=1,
        )

        # ── 5. Save final model ────────────────────────────────
        final_path = Path(self.cfg["paths"]["model_save_dir"]) / "cnn_lstm_final.h5"
        model.save(str(final_path))
        logger.info(f"[5/5] Final model saved → {final_path}")

        # ── 6. Plot training curves ────────────────────────────
        self._plot_history(history)

        return model

    def _plot_history(self, history: tf.keras.callbacks.History) -> None:
        """Save training/validation accuracy and loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Training History — CNN-LSTM Congestion Model",
                     fontsize=14, fontweight="bold")

        # Accuracy
        axes[0].plot(history.history.get("accuracy", []),
                     label="Train Accuracy", color="#2196F3", linewidth=2)
        axes[0].plot(history.history.get("val_accuracy", []),
                     label="Val Accuracy",   color="#FF5722", linewidth=2,
                     linestyle="--")
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(history.history.get("loss", []),
                     label="Train Loss", color="#4CAF50", linewidth=2)
        axes[1].plot(history.history.get("val_loss", []),
                     label="Val Loss",   color="#9C27B0", linewidth=2,
                     linestyle="--")
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out = Path(self.cfg["paths"]["report_dir"]) / "training_history.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Training curve saved → {out}")


# ── CLI entry point ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CNN-LSTM congestion model")
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--use_augmented", action="store_true",
                        help="Train on augmented dataset")
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--lr",            type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.epochs     is not None: cfg["training"]["epochs"]        = args.epochs
    if args.batch_size is not None: cfg["training"]["batch_size"]    = args.batch_size
    if args.lr         is not None: cfg["training"]["learning_rate"] = args.lr

    trainer = Trainer(cfg, use_augmented=args.use_augmented)
    trainer.run()
