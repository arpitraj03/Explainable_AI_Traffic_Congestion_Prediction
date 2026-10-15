"""
run_pipeline.py
===============
End-to-End Pipeline Runner

Runs all stages in sequence:
  Stage 1: Video preprocessing → extract frame sequences
  Stage 2: Data augmentation   → expand training data
  Stage 3: Model training      → CNN-LSTM training loop
  Stage 4: Model evaluation    → metrics, confusion matrix
  Stage 5: XAI explanations    → Grad-CAM, SHAP, LIME

Usage:
    python run_pipeline.py
    python run_pipeline.py --video_dir data/raw/ --epochs 30 --batch_size 8
    python run_pipeline.py --skip_augment    # skip augmentation stage
    python run_pipeline.py --stage eval      # run from evaluation onwards
"""

import sys
import os
import yaml
import logging
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing  import VideoDatasetBuilder
from augmentation   import DatasetAugmentor
from dataset        import DataManager
from model          import CongestionModel, build_callbacks
from train          import Trainer
from evaluate       import Evaluator
from explainability import XAIRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

STAGES = ["preprocess", "augment", "train", "eval", "xai"]


def banner(title: str) -> None:
    w = 60
    print("\n" + "═" * w)
    print(f"  {title}")
    print("═" * w)


def run_pipeline(cfg: dict, args) -> None:
    """Execute the full or partial pipeline."""

    start_stage = STAGES.index(args.stage) if args.stage in STAGES else 0

    # ──────────────────────────────────────────────────────────
    # STAGE 1: Preprocessing
    # ──────────────────────────────────────────────────────────
    if start_stage <= 0:
        banner("STAGE 1 / 5  —  VIDEO PREPROCESSING")
        raw_dir  = args.video_dir or cfg["paths"]["raw_video_dir"]
        proc_dir = cfg["paths"]["processed_dir"]

        builder = VideoDatasetBuilder(raw_dir, proc_dir, cfg)
        X, y = builder.build()

        class_counts = {
            k: int((y == v).sum())
            for k, v in cfg["dataset"]["class_to_idx"].items()
        }
        logger.info(f"Processed dataset shape: X{X.shape}, y{y.shape}")
        logger.info(f"Class distribution: {class_counts}")
    else:
        logger.info("Skipping preprocessing stage.")

    # ──────────────────────────────────────────────────────────
    # STAGE 2: Augmentation
    # ──────────────────────────────────────────────────────────
    if start_stage <= 1 and not args.skip_augment:
        banner("STAGE 2 / 5  —  DATA AUGMENTATION")
        aug = DatasetAugmentor(
            processed_dir=cfg["paths"]["processed_dir"],
            augmented_dir=cfg["paths"]["augmented_dir"],
            cfg=cfg,
            copies=2,
        )
        aug.run()
    else:
        logger.info("Skipping augmentation stage.")

    # ──────────────────────────────────────────────────────────
    # STAGE 3: Training
    # ──────────────────────────────────────────────────────────
    if start_stage <= 2:
        banner("STAGE 3 / 5  —  MODEL TRAINING")

        # Apply CLI overrides to config
        if args.epochs:     cfg["training"]["epochs"]        = args.epochs
        if args.batch_size: cfg["training"]["batch_size"]    = args.batch_size
        if args.lr:         cfg["training"]["learning_rate"] = args.lr

        use_aug = (not args.skip_augment and
                   Path(cfg["paths"]["augmented_dir"]).exists() and
                   (Path(cfg["paths"]["augmented_dir"]) / "augmented_X.npy").exists())

        trainer = Trainer(cfg, use_augmented=use_aug)
        model   = trainer.run()
        logger.info("Training complete.")
    else:
        logger.info("Loading model for eval/XAI stages...")
        model_path = cfg["paths"]["model_checkpoint"]
        if not Path(model_path).exists():
            logger.error(
                f"No model found at {model_path}. "
                "Run training first (remove --stage eval/xai)."
            )
            sys.exit(1)
        model = tf.keras.models.load_model(model_path)

    # ──────────────────────────────────────────────────────────
    # STAGE 4: Evaluation
    # ──────────────────────────────────────────────────────────
    if start_stage <= 3:
        banner("STAGE 4 / 5  —  MODEL EVALUATION")
        evaluator = Evaluator(cfg["paths"]["model_checkpoint"], cfg)
        metrics   = evaluator.evaluate()

        print("\n  ── Final Metrics ──────────────────────────")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  Macro P   : {metrics['macro_p']:.4f}")
        print(f"  Macro R   : {metrics['macro_r']:.4f}")
        print(f"  Macro F1  : {metrics['macro_f1']:.4f}")
        print("  ─────────────────────────────────────────\n")
    else:
        logger.info("Skipping evaluation stage.")

    # ──────────────────────────────────────────────────────────
    # STAGE 5: Explainability (XAI)
    # ──────────────────────────────────────────────────────────
    if start_stage <= 4 and not args.skip_xai:
        banner("STAGE 5 / 5  —  EXPLAINABILITY (Grad-CAM / SHAP / LIME)")

        # Load test sequences for XAI
        dm   = DataManager(cfg)
        data = dm.prepare()
        X_te = data["X_test"]
        y_te = data["y_test"]

        xai_runner = XAIRunner(model, cfg)
        xai_runner.run(X_te, y_te, max_samples=5)
    else:
        logger.info("Skipping XAI stage.")

    banner("PIPELINE COMPLETE")
    print(f"  Outputs saved to : {cfg['paths']['report_dir']}")
    print(f"  Heatmaps saved to: {cfg['paths']['heatmap_dir']}")
    print(f"  Model saved to   : {cfg['paths']['model_checkpoint']}")
    print()
    print("  Next steps:")
    print("  • Run inference  : python src/predict.py --video <your_video.mp4>")
    print("  • Launch dashboard: streamlit run dashboard/app.py")
    print()


# ── CLI ────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Explainable AI Traffic Congestion Prediction Pipeline"
    )
    parser.add_argument(
        "--config",       default="config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--video_dir",    default=None,
        help="Override path to raw CCTV video directory"
    )
    parser.add_argument(
        "--stage",        default="preprocess",
        choices=STAGES,
        help="Start pipeline from this stage"
    )
    parser.add_argument(
        "--epochs",       type=int,   default=None,
        help="Override training epochs"
    )
    parser.add_argument(
        "--batch_size",   type=int,   default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--lr",           type=float, default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--skip_augment", action="store_true",
        help="Skip data augmentation stage"
    )
    parser.add_argument(
        "--skip_xai",     action="store_true",
        help="Skip XAI explanation stage"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Ensure output dirs exist
    for key in ["model_save_dir", "processed_dir", "augmented_dir",
                "output_vis_dir", "heatmap_dir", "report_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    run_pipeline(cfg, args)