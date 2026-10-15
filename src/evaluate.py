"""
evaluate.py
===========
Model Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix

Loads the best-checkpoint model and evaluates it on the held-out
test set, producing:
  - Per-class precision, recall, F1-score
  - Macro-averaged summary
  - Confusion matrix heatmap
  - Per-video congestion timeline plots
"""

import sys
import yaml
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).parent))
from dataset import DataManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


# ══════════════════════════════════════════════════════════════
#  Evaluator
# ══════════════════════════════════════════════════════════════
class Evaluator:
    """
    Evaluate a trained model on the test split.

    Args:
        model_path : Path to the saved Keras .h5 model.
        cfg        : Full config dict.
    """

    CLASS_NAMES = ["Low", "Medium", "High"]
    COLORS      = ["#4CAF50", "#FF9800", "#F44336"]

    def __init__(self, model_path: str, cfg: dict):
        self.cfg        = cfg
        self.report_dir = Path(cfg["paths"]["report_dir"])
        self.report_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(model_path)

    def evaluate(self) -> dict:
        """
        Run full evaluation on the test set.

        Returns:
            Dictionary of metrics (accuracy, per-class P/R/F1, macro averages).
        """
        # ── Load test data ─────────────────────────────────────
        dm   = DataManager(self.cfg)
        data = dm.prepare()
        X_te = data["X_test"]
        y_te = data["y_test"]

        logger.info(f"Evaluating on {len(X_te)} test sequences...")

        # ── Predictions ────────────────────────────────────────
        y_proba = self.model.predict(X_te, batch_size=8, verbose=0)
        y_pred  = np.argmax(y_proba, axis=1)

        # ── Metrics ────────────────────────────────────────────
        acc = accuracy_score(y_te, y_pred)
        prec, rec, f1, support = precision_recall_fscore_support(
            y_te, y_pred, average=None, labels=[0, 1, 2]
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_te, y_pred, average="macro"
        )

        report_str = classification_report(
            y_te, y_pred,
            target_names=self.CLASS_NAMES,
            digits=4
        )

        # ── Log results ────────────────────────────────────────
        logger.info("\n" + "=" * 55)
        logger.info("  EVALUATION RESULTS")
        logger.info("=" * 55)
        logger.info(f"  Overall Accuracy : {acc:.4f}")
        logger.info(f"  Macro Precision  : {macro_p:.4f}")
        logger.info(f"  Macro Recall     : {macro_r:.4f}")
        logger.info(f"  Macro F1-Score   : {macro_f1:.4f}")
        logger.info("\n" + report_str)

        # Save classification report to file
        rpt_path = self.report_dir / "classification_report.txt"
        with open(rpt_path, "w") as f:
            f.write("=" * 55 + "\n")
            f.write("  CNN-LSTM TRAFFIC CONGESTION EVALUATION REPORT\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"Overall Accuracy : {acc:.4f}\n")
            f.write(f"Macro Precision  : {macro_p:.4f}\n")
            f.write(f"Macro Recall     : {macro_r:.4f}\n")
            f.write(f"Macro F1-Score   : {macro_f1:.4f}\n\n")
            f.write(report_str)
        logger.info(f"Saved report → {rpt_path}")

        # ── Visualisations ─────────────────────────────────────
        self._plot_confusion_matrix(y_te, y_pred)
        self._plot_per_class_metrics(prec, rec, f1)
        self._plot_prediction_confidence(y_proba, y_te)

        return {
            "accuracy":    acc,
            "precision":   prec.tolist(),
            "recall":      rec.tolist(),
            "f1":          f1.tolist(),
            "macro_p":     macro_p,
            "macro_r":     macro_r,
            "macro_f1":    macro_f1,
            "y_true":      y_te,
            "y_pred":      y_pred,
            "y_proba":     y_proba,
        }

    def _plot_confusion_matrix(self, y_true: np.ndarray,
                               y_pred: np.ndarray) -> None:
        """Save a colour-coded confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Confusion Matrix — CNN-LSTM Congestion Model",
                     fontsize=13, fontweight="bold")

        for ax, data, fmt, title in zip(
            axes,
            [cm, cm_norm],
            ["d", ".2%"],
            ["Raw Counts", "Normalised (Row %)"]
        ):
            sns.heatmap(
                data, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=self.CLASS_NAMES,
                yticklabels=self.CLASS_NAMES,
                ax=ax, linewidths=0.5
            )
            ax.set_title(title)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

        plt.tight_layout()
        out = self.report_dir / "confusion_matrix.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved → {out}")

    def _plot_per_class_metrics(self, prec: np.ndarray,
                                rec: np.ndarray, f1: np.ndarray) -> None:
        """Bar chart of per-class Precision / Recall / F1."""
        x = np.arange(len(self.CLASS_NAMES))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width, prec, width, label="Precision", color="#2196F3")
        ax.bar(x,         rec,  width, label="Recall",    color="#4CAF50")
        ax.bar(x + width, f1,   width, label="F1-Score",  color="#FF9800")

        ax.set_xticks(x)
        ax.set_xticklabels(self.CLASS_NAMES)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Evaluation Metrics", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for i, (p, r, f) in enumerate(zip(prec, rec, f1)):
            ax.text(i - width, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
            ax.text(i,         r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
            ax.text(i + width, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)

        plt.tight_layout()
        out = self.report_dir / "per_class_metrics.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Per-class metrics chart saved → {out}")

    def _plot_prediction_confidence(self, y_proba: np.ndarray,
                                    y_true: np.ndarray) -> None:
        """Box-plot of prediction confidence stratified by true class."""
        fig, ax = plt.subplots(figsize=(10, 5))

        for cls_idx, cls_name in enumerate(self.CLASS_NAMES):
            mask  = y_true == cls_idx
            confs = y_proba[mask, cls_idx]
            ax.boxplot(confs, positions=[cls_idx], widths=0.5,
                       patch_artist=True,
                       boxprops=dict(facecolor=self.COLORS[cls_idx], alpha=0.6))

        ax.set_xticks(range(len(self.CLASS_NAMES)))
        ax.set_xticklabels(self.CLASS_NAMES)
        ax.set_ylabel("Softmax Confidence")
        ax.set_title("Prediction Confidence by True Class", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out = self.report_dir / "confidence_boxplot.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Confidence plot saved → {out}")


# ── CLI entry point ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--model_path", default=None,
                        help="Override model checkpoint path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path = args.model_path or cfg["paths"]["model_checkpoint"]
    ev = Evaluator(model_path, cfg)
    metrics = ev.evaluate()
