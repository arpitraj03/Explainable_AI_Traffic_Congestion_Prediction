"""
explainability.py
=================
Explainable AI for Traffic Congestion Predictions

Three complementary XAI techniques:

1. Grad-CAM  — Gradient-weighted Class Activation Mapping
   Computes the gradient of the predicted class score with respect
   to the last convolutional feature map, then produces a heatmap
   highlighting which spatial regions drove the prediction.
   → Best for: "Where in the frame did the model look?"

2. SHAP (KernelSHAP)
   Treats the model as a black box and computes Shapley values for
   super-pixel regions by sampling coalitions of masked inputs.
   → Best for: "How much did each image region contribute?"

3. LIME  — Local Interpretable Model-agnostic Explanations
   Fits a locally linear surrogate model by perturbing superpixels
   around the input frame.
   → Best for: "Which regions most change the prediction locally?"

All methods return overlaid visualisations saved to disk.
"""

import sys
import logging
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from pathlib import Path
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

CLASS_NAMES = ["Low", "Medium", "High"]


# ══════════════════════════════════════════════════════════════
#  GradCAM Explainer
# ══════════════════════════════════════════════════════════════
class GradCAMExplainer:
    """
    Grad-CAM for the CNN-LSTM hybrid model.

    Since the CNN is wrapped in TimeDistributed, we extract the CNN
    sub-model and apply Grad-CAM to individual frames.

    Args:
        full_model    : The full CNN-LSTM Keras model.
        target_layer  : Name of the last convolutional layer in the CNN.
        colormap      : Matplotlib colormap name for the heatmap.
        alpha         : Transparency for heatmap overlay (0=invisible, 1=opaque).
    """

    def __init__(self, full_model: tf.keras.Model,
                 target_layer: str = "block_16_project_BN",
                 colormap: str = "jet",
                 alpha: float = 0.4):
        self.full_model   = full_model
        self.target_layer = target_layer
        self.colormap     = colormap
        self.alpha        = alpha

        # Extract the inner CNN model from TimeDistributed wrapper
        self.cnn_model = self._extract_cnn(full_model)
        self.grad_model = self._build_grad_model()

    def _extract_cnn(self, model: tf.keras.Model) -> Optional[tf.keras.Model]:
        """Dig out the CNN sub-model from the TimeDistributed layer."""
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.TimeDistributed):
                return layer.layer   # inner model
        logger.warning("No TimeDistributed layer found; using full model as CNN.")
        return model

    def _build_grad_model(self) -> tf.keras.Model:
        """
        Build a model that outputs both the target layer activations
        and the final predictions.
        """
        if self.cnn_model is None:
            return None

        try:
            target_layer_obj = self.cnn_model.get_layer(self.target_layer)
        except ValueError:
            # Fallback: use the last Conv2D layer found
            for layer in reversed(self.cnn_model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    target_layer_obj = layer
                    self.target_layer = layer.name
                    logger.info(f"Grad-CAM target layer set to: {layer.name}")
                    break
            else:
                logger.error("No Conv2D layer found for Grad-CAM.")
                return None

        return tf.keras.Model(
            inputs=self.cnn_model.inputs,
            outputs=[target_layer_obj.output, self.cnn_model.output],
            name="grad_cam_model"
        )

    def compute_heatmap(self, frame: np.ndarray,
                        class_idx: Optional[int] = None) -> np.ndarray:
        """
        Compute a Grad-CAM heatmap for a single frame.

        Args:
            frame     : Float32 array (H, W, 3) in [0, 1].
            class_idx : Target class index. If None, uses the predicted class.

        Returns:
            Heatmap as uint8 BGR image (H, W, 3) with values in [0, 255].
        """
        if self.grad_model is None:
            return np.zeros((*frame.shape[:2], 3), dtype=np.uint8)

        img_batch = np.expand_dims(frame, axis=0)   # (1, H, W, 3)

        with tf.GradientTape() as tape:
            inputs    = tf.cast(img_batch, tf.float32)
            tape.watch(inputs)
            conv_outs, preds = self.grad_model(inputs)
            if class_idx is None:
                class_idx = tf.argmax(preds[0]).numpy()
            score = preds[:, class_idx]

        grads     = tape.gradient(score, conv_outs)          # (1, h, w, c)
        pooled    = tf.reduce_mean(grads, axis=(1, 2))       # (1, c)
        conv_outs = conv_outs[0]                              # (h, w, c)
        pooled    = pooled[0]                                 # (c,)

        # Weighted combination of feature maps
        cam = tf.reduce_sum(
            tf.cast(conv_outs, tf.float32) *
            tf.cast(pooled[:, tf.newaxis, tf.newaxis], tf.float32)   # broadcast
            , axis=-1
        ).numpy()

        # Normalise and resize to frame dimensions
        cam = np.maximum(cam, 0)                                  # ReLU
        if cam.max() > 0:
            cam = cam / cam.max()

        H, W = frame.shape[:2]
        cam_resized = cv2.resize(cam, (W, H))

        # Apply colour map
        heatmap_uint8 = np.uint8(255 * cam_resized)
        heatmap_colour = cv2.applyColorMap(heatmap_uint8,
                                           getattr(cv2, f"COLORMAP_{self.colormap.upper()}",
                                                   cv2.COLORMAP_JET))
        return heatmap_colour

    def overlay(self, frame: np.ndarray,
                class_idx: Optional[int] = None) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original frame.

        Returns:
            BGR uint8 overlay image (H, W, 3).
        """
        frame_uint8 = np.uint8(frame * 255) if frame.max() <= 1.0 else frame
        frame_bgr   = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        heatmap     = self.compute_heatmap(frame, class_idx)
        overlay     = cv2.addWeighted(frame_bgr, 1 - self.alpha,
                                      heatmap,    self.alpha, 0)
        return overlay

    def explain_sequence(self, sequence: np.ndarray,
                         pred_class: int,
                         save_dir: str,
                         sample_idx: int = 0,
                         max_frames: int = 4) -> None:
        """
        Generate and save Grad-CAM overlays for key frames in a sequence.

        Args:
            sequence  : Float32 (T, H, W, 3).
            pred_class: Predicted class index.
            save_dir  : Directory to save visualisation images.
            sample_idx: Sample number (for filename).
            max_frames: How many frames to visualise.
        """
        T = sequence.shape[0]
        frame_indices = np.linspace(0, T - 1, min(max_frames, T), dtype=int)

        fig, axes = plt.subplots(2, len(frame_indices),
                                 figsize=(4 * len(frame_indices), 8))
        if len(frame_indices) == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle(
            f"Grad-CAM Explanation  |  Predicted: {CLASS_NAMES[pred_class]}  "
            f"Congestion",
            fontsize=13, fontweight="bold"
        )

        for col, t in enumerate(frame_indices):
            frame   = sequence[t]                             # (H, W, 3)
            overlay = self.overlay(frame, pred_class)        # BGR uint8
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            axes[0, col].imshow(frame)
            axes[0, col].set_title(f"Frame {t}")
            axes[0, col].axis("off")

            axes[1, col].imshow(overlay_rgb)
            axes[1, col].set_title("Grad-CAM")
            axes[1, col].axis("off")

        plt.tight_layout()
        out = Path(save_dir) / f"gradcam_sample_{sample_idx}.png"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Grad-CAM saved → {out}")


# ══════════════════════════════════════════════════════════════
#  SHAP Explainer (KernelSHAP on spatial segments)
# ══════════════════════════════════════════════════════════════
class SHAPExplainer:
    """
    KernelSHAP explainer for a single representative frame.

    KernelSHAP masks super-pixel regions and measures how each
    region contributes to the predicted class score.

    Note: We explain the CNN sub-model on individual frames
    (the middle frame of the sequence) for interpretability.

    Args:
        cnn_model       : CNN sub-model (H, W, 3) → softmax scores.
        num_background  : Number of background samples for KernelSHAP.
    """

    def __init__(self, cnn_model: tf.keras.Model,
                 num_background: int = 50):
        try:
            import shap
            self.shap = shap
        except ImportError:
            logger.warning("SHAP not installed. Run: pip install shap")
            self.shap = None

        self.cnn_model     = cnn_model
        self.num_background = num_background
        self.explainer     = None

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        """Prediction function for SHAP: (N, H, W, 3) → (N, num_classes)."""
        return self.cnn_model.predict(X, verbose=0)

    def explain_frame(self, frame: np.ndarray,
                      background_frames: np.ndarray,
                      save_dir: str,
                      sample_idx: int = 0) -> None:
        """
        Compute and save SHAP values for a single frame.

        Args:
            frame            : Float32 (H, W, 3).
            background_frames: Float32 (N, H, W, 3) background dataset.
            save_dir         : Output directory.
            sample_idx       : Index for filename.
        """
        if self.shap is None:
            logger.warning("Skipping SHAP explanation (shap not installed).")
            return

        # Sample background
        bg = background_frames[:self.num_background]
        explainer = self.shap.KernelExplainer(
            self._predict_fn, bg
        )

        shap_vals = explainer.shap_values(
            np.expand_dims(frame, axis=0),
            nsamples=200,
            silent=True
        )  # List of (1, H, W, 3) arrays, one per class

        # Summarise per class by averaging over pixels
        num_classes = len(shap_vals)
        fig, axes = plt.subplots(1, num_classes + 1,
                                 figsize=(5 * (num_classes + 1), 5))
        fig.suptitle(f"SHAP Feature Importance — Sample {sample_idx}",
                     fontsize=13, fontweight="bold")

        axes[0].imshow(frame)
        axes[0].set_title("Original Frame")
        axes[0].axis("off")

        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            shap_map = shap_vals[cls_idx][0]         # (H, W, 3)
            shap_abs = np.abs(shap_map).mean(axis=-1) # (H, W)
            axes[cls_idx + 1].imshow(frame, alpha=0.5)
            im = axes[cls_idx + 1].imshow(shap_abs, cmap="hot", alpha=0.6)
            axes[cls_idx + 1].set_title(f"SHAP: {cls_name}")
            axes[cls_idx + 1].axis("off")
            plt.colorbar(im, ax=axes[cls_idx + 1], fraction=0.046)

        plt.tight_layout()
        out = Path(save_dir) / f"shap_sample_{sample_idx}.png"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP explanation saved → {out}")


# ══════════════════════════════════════════════════════════════
#  LIME Explainer
# ══════════════════════════════════════════════════════════════
class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for
    a single representative frame.

    Segments the frame into superpixels using SLIC, then perturbs
    subsets of segments and fits a locally linear model.

    Args:
        cnn_model   : CNN sub-model (H, W, 3) → softmax scores.
        num_samples : Number of perturbation samples.
        num_segments: Number of SLIC superpixels.
        num_features: Number of top features to highlight.
    """

    def __init__(self, cnn_model: tf.keras.Model,
                 num_samples: int = 1000,
                 num_segments: int = 50,
                 num_features: int = 10):
        try:
            from lime import lime_image
            self.lime_image = lime_image
        except ImportError:
            logger.warning("LIME not installed. Run: pip install lime")
            self.lime_image = None

        self.cnn_model    = cnn_model
        self.num_samples  = num_samples
        self.num_segments = num_segments
        self.num_features = num_features

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        """Prediction function for LIME: (N, H, W, 3) → (N, num_classes)."""
        return self.cnn_model.predict(X, verbose=0)

    def explain_frame(self, frame: np.ndarray,
                      pred_class: int,
                      save_dir: str,
                      sample_idx: int = 0) -> None:
        """
        Compute and save LIME explanation for a single frame.

        Args:
            frame      : Float32 (H, W, 3).
            pred_class : Predicted class index.
            save_dir   : Output directory.
            sample_idx : Index for filename.
        """
        if self.lime_image is None:
            logger.warning("Skipping LIME explanation (lime not installed).")
            return

        explainer = self.lime_image.LimeImageExplainer()

        # Convert to uint8 for LIME (expects 0-255 range)
        frame_uint8 = (frame * 255).astype(np.uint8)

        def predict_wrapper(X):
            return self._predict_fn(X.astype(np.float32) / 255.0)

        explanation = explainer.explain_instance(
            frame_uint8,
            predict_wrapper,
            top_labels=3,
            hide_color=0,
            num_samples=self.num_samples,
            num_features=self.num_features,
        )

        # Get image + mask for predicted class
        temp, mask = explanation.get_image_and_mask(
            pred_class,
            positive_only=False,
            num_features=self.num_features,
            hide_rest=False,
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"LIME Explanation  |  Predicted: {CLASS_NAMES[pred_class]} Congestion  "
            f"| Sample {sample_idx}",
            fontsize=13, fontweight="bold"
        )

        axes[0].imshow(frame)
        axes[0].set_title("Original Frame")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[1].set_title("LIME Segment Mask\n(Green=+ve, Red=-ve)")
        axes[1].axis("off")

        axes[2].imshow(temp.astype(np.uint8))
        axes[2].set_title("Frame with LIME Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        out = Path(save_dir) / f"lime_sample_{sample_idx}.png"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"LIME explanation saved → {out}")


# ══════════════════════════════════════════════════════════════
#  XAIRunner  (orchestrates all three methods)
# ══════════════════════════════════════════════════════════════
class XAIRunner:
    """
    Convenience class that runs all enabled XAI methods on a batch
    of test sequences.

    Args:
        model   : Full trained CNN-LSTM model.
        cfg     : Full config dict.
    """

    def __init__(self, model: tf.keras.Model, cfg: dict):
        self.model   = model
        self.cfg     = cfg
        self.xai_cfg = cfg["explainability"]
        self.heatmap_dir = cfg["paths"]["heatmap_dir"]

        # Extract CNN sub-model for frame-level explanations
        self.cnn_model = self._get_cnn()

    def _get_cnn(self) -> Optional[tf.keras.Model]:
        """Extract CNN from TimeDistributed wrapper."""
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.TimeDistributed):
                return layer.layer
        return None

    def run(self, sequences: np.ndarray,
            y_true: np.ndarray,
            max_samples: int = 5) -> None:
        """
        Run all enabled XAI methods on a subset of test sequences.

        Args:
            sequences  : Float32 (N, T, H, W, 3).
            y_true     : Int array (N,).
            max_samples: Max number of samples to explain.
        """
        n = min(max_samples, len(sequences))
        logger.info(f"Running XAI on {n} samples...")

        # Predict classes
        y_proba = self.model.predict(sequences[:n], verbose=0)
        y_pred  = np.argmax(y_proba, axis=1)

        for i in range(n):
            seq       = sequences[i]   # (T, H, W, 3)
            pred_cls  = y_pred[i]
            mid_frame = seq[seq.shape[0] // 2]   # representative middle frame

            # ── Grad-CAM ───────────────────────────────────────
            if self.xai_cfg["gradcam"]["enabled"]:
                gc = GradCAMExplainer(
                    full_model=self.model,
                    target_layer=self.xai_cfg["gradcam"]["target_layer"],
                    colormap=self.xai_cfg["gradcam"]["colormap"],
                    alpha=self.xai_cfg["gradcam"]["alpha"],
                )
                gc.explain_sequence(seq, pred_cls, self.heatmap_dir, i)

            # ── SHAP ───────────────────────────────────────────
            if self.xai_cfg["shap"]["enabled"] and self.cnn_model is not None:
                shap_exp = SHAPExplainer(
                    cnn_model=self.cnn_model,
                    num_background=self.xai_cfg["shap"]["num_background"],
                )
                bg_frames = sequences[:self.xai_cfg["shap"]["num_background"],
                                      seq.shape[0] // 2]
                shap_exp.explain_frame(mid_frame, bg_frames,
                                       self.heatmap_dir, i)

            # ── LIME ───────────────────────────────────────────
            if self.xai_cfg["lime"]["enabled"] and self.cnn_model is not None:
                lime_exp = LIMEExplainer(
                    cnn_model=self.cnn_model,
                    num_samples=self.xai_cfg["lime"]["num_samples"],
                    num_segments=self.xai_cfg["lime"]["num_segments"],
                    num_features=self.xai_cfg["lime"]["num_features"],
                )
                lime_exp.explain_frame(mid_frame, pred_cls,
                                       self.heatmap_dir, i)

        logger.info(f"All XAI explanations saved to {self.heatmap_dir}")
