"""
predict.py
==========
Real-Time Inference on New CCTV Video Inputs

Given a new .mp4 / .avi video file, this script:
  1. Extracts and preprocesses frame sequences
  2. Loads the trained CNN-LSTM model
  3. Predicts the congestion level per sequence
  4. Generates Grad-CAM heatmaps for every sequence
  5. Writes an annotated output video
  6. Plots the full congestion timeline
  7. Prints a per-second summary to stdout

Usage:
    python src/predict.py --video path/to/cctv.mp4
    python src/predict.py --video path/to/cctv.mp4 --model models/cnn_lstm_best.h5
"""

import sys
import os
import yaml
import logging
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing  import FrameExtractor
from explainability import GradCAMExplainer
from visualizer     import FrameAnnotator, TimelinePlotter, AnnotatedVideoWriter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["Low", "Medium", "High"]
CLASS_ICONS = {0: "🟢", 1: "🟡", 2: "🔴"}


# ══════════════════════════════════════════════════════════════
#  Predictor
# ══════════════════════════════════════════════════════════════
class Predictor:
    """
    End-to-end inference on a single CCTV video.

    Args:
        model_path : Path to saved .h5 model.
        cfg        : Full config dict.
    """

    def __init__(self, model_path: str, cfg: dict):
        self.cfg      = cfg
        self.out_vis  = Path(cfg["paths"]["output_vis_dir"])
        self.hm_dir   = Path(cfg["paths"]["heatmap_dir"])
        self.out_vis.mkdir(parents=True, exist_ok=True)
        self.hm_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading model: {model_path}")
        self.model = tf.keras.models.load_model(model_path)

        self.extractor = FrameExtractor(
            height=cfg["preprocessing"]["frame_height"],
            width=cfg["preprocessing"]["frame_width"],
            sequence_length=cfg["preprocessing"]["sequence_length"],
            frame_skip=cfg["preprocessing"]["frame_skip"],
            use_bg_sub=cfg["preprocessing"]["background_subtraction"],
            normalize=cfg["preprocessing"]["normalize"],
        )

        self.grad_cam = GradCAMExplainer(
            full_model=self.model,
            target_layer=cfg["explainability"]["gradcam"]["target_layer"],
            colormap=cfg["explainability"]["gradcam"]["colormap"],
            alpha=cfg["explainability"]["gradcam"]["alpha"],
        ) if cfg["explainability"]["gradcam"]["enabled"] else None

    def predict_video(self, video_path: str,
                      output_video: bool = True,
                      run_gradcam: bool = True) -> dict:
        """
        Full inference pipeline for a single video.

        Returns:
            Dictionary with keys:
              video_path, predictions, confidences, probabilities,
              fps, sequence_length, output_video_path
        """
        video_path = str(video_path)
        video_name = Path(video_path).name
        logger.info(f"Processing video: {video_name}")

        # ── 1. Extract sequences ────────────────────────────────
        sequences = self.extractor.extract_sequences(video_path)
        fps       = self.extractor.get_video_fps(video_path)
        meta      = self.extractor.get_video_metadata(video_path)

        if not sequences:
            logger.error(f"No sequences extracted from {video_path}")
            return {}

        X = np.stack(sequences, axis=0)    # (N, T, H, W, 3)
        logger.info(f"Extracted {len(X)} sequences from {video_name}")

        # ── 2. Predict ─────────────────────────────────────────
        y_proba       = self.model.predict(X, batch_size=4, verbose=0)
        y_pred        = np.argmax(y_proba, axis=1)
        confidences   = y_proba.max(axis=1)

        # ── 3. Console summary ─────────────────────────────────
        self._print_summary(video_name, y_pred, confidences, fps, meta)

        # ── 4. Annotated output video ──────────────────────────
        out_vid_path = None
        if output_video:
            out_vid_path = str(self.out_vis / f"annotated_{Path(video_path).stem}.mp4")
            H = self.cfg["preprocessing"]["frame_height"]
            W = self.cfg["preprocessing"]["frame_width"]
            writer = AnnotatedVideoWriter(
                output_path=out_vid_path,
                fps=max(fps / self.cfg["preprocessing"]["frame_skip"], 5),
                frame_size=(W, H)
            )
            for i, seq in enumerate(sequences):
                writer.write_sequence(seq, int(y_pred[i]), float(confidences[i]))
            writer.release()

        # ── 5. Congestion timeline ─────────────────────────────
        TimelinePlotter.plot(
            predictions=y_pred.tolist(),
            confidences=confidences.tolist(),
            video_name=video_name,
            save_dir=str(self.out_vis),
            fps=fps,
            frame_skip=self.cfg["preprocessing"]["frame_skip"],
            sequence_length=self.cfg["preprocessing"]["sequence_length"],
        )

        # ── 6. Grad-CAM heatmaps ───────────────────────────────
        if run_gradcam and self.grad_cam is not None:
            num_explain = min(5, len(sequences))
            for i in range(num_explain):
                self.grad_cam.explain_sequence(
                    sequence=sequences[i],
                    pred_class=int(y_pred[i]),
                    save_dir=str(self.hm_dir),
                    sample_idx=i,
                )

        return {
            "video_path":        video_path,
            "predictions":       y_pred.tolist(),
            "confidences":       confidences.tolist(),
            "probabilities":     y_proba.tolist(),
            "fps":               fps,
            "sequence_length":   self.cfg["preprocessing"]["sequence_length"],
            "output_video_path": out_vid_path,
            "metadata":          meta,
        }

    def _print_summary(self, video_name: str,
                       y_pred: np.ndarray,
                       confidences: np.ndarray,
                       fps: float,
                       meta: dict) -> None:
        """Print a human-readable congestion summary to the console."""
        secs_per_seq = (
            self.cfg["preprocessing"]["sequence_length"] *
            self.cfg["preprocessing"]["frame_skip"]
        ) / max(fps, 1)

        print("\n" + "=" * 65)
        print(f"  CONGESTION PREDICTION REPORT: {video_name}")
        print("=" * 65)
        print(f"  Duration : {meta.get('duration_s', '?'):.1f}s  |  "
              f"Resolution: {meta.get('width')}×{meta.get('height')}  |  "
              f"FPS: {fps:.1f}")
        print(f"  Sequences: {len(y_pred)}")
        print("-" * 65)
        print(f"  {'Seq':>4}  {'Time':>8}  {'Level':>8}  {'Conf':>7}  {'Bar'}")
        print("-" * 65)

        for i, (pred, conf) in enumerate(zip(y_pred, confidences)):
            t_start = i * secs_per_seq
            bar_len = int(conf * 20)
            bar     = "█" * bar_len + "░" * (20 - bar_len)
            icon    = CLASS_ICONS[int(pred)]
            print(f"  {i:>4}  {t_start:>6.1f}s  "
                  f"{icon} {CLASS_NAMES[int(pred)]:>6}  "
                  f"{conf:>6.1%}  {bar}")

        # Summary stats
        from collections import Counter
        counts = Counter(y_pred.tolist())
        total  = len(y_pred)
        print("-" * 65)
        print("  SUMMARY:")
        for cls_idx, name in enumerate(CLASS_NAMES):
            pct = counts.get(cls_idx, 0) / total * 100
            icon = CLASS_ICONS[cls_idx]
            print(f"    {icon} {name:>6}: {counts.get(cls_idx, 0):>3} seqs ({pct:.1f}%)")
        dominant = CLASS_NAMES[int(max(counts, key=counts.get))]
        print(f"  Overall Assessment: {CLASS_ICONS[CLASS_NAMES.index(dominant)]} "
              f"{dominant.upper()} CONGESTION")
        print("=" * 65 + "\n")


# ── CLI ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Predict traffic congestion on a CCTV video"
    )
    parser.add_argument("--video",      required=True,
                        help="Path to input CCTV video (.mp4 / .avi)")
    parser.add_argument("--model",      default=None,
                        help="Path to model .h5 (overrides config)")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--no_video",   action="store_true",
                        help="Skip annotated video output")
    parser.add_argument("--no_gradcam", action="store_true",
                        help="Skip Grad-CAM heatmap generation")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path = args.model or cfg["paths"]["model_checkpoint"]

    # Check model exists; if not, inform user
    if not Path(model_path).exists():
        logger.warning(
            f"Model not found at {model_path}.\n"
            "Please train the model first:\n"
            "  python src/train.py --config config.yaml\n"
            "Or run the full pipeline:\n"
            "  python run_pipeline.py\n"
        )
        sys.exit(1)

    predictor = Predictor(model_path, cfg)
    results   = predictor.predict_video(
        video_path=args.video,
        output_video=not args.no_video,
        run_gradcam=not args.no_gradcam,
    )

    if results:
        logger.info("Inference complete.")
        logger.info(f"Outputs saved to: {cfg['paths']['output_vis_dir']}")


if __name__ == "__main__":
    main()
