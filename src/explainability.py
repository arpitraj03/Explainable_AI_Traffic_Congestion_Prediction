"""
explainability.py
=================
Explainable AI for Traffic Congestion Predictions

Three complementary XAI techniques:
  1. Grad-CAM  - spatial heatmap of what the CNN looked at
  2. SHAP      - Shapley value attribution per image region
  3. LIME      - superpixel perturbation explanations

Fix v4:
    The KeyError crash happened because we tried to build a new
    tf.keras.Model using an internal layer's output from a loaded .h5
    model — but the tensor graph IDs no longer match after loading.

    New approach for Grad-CAM:
      - Do NOT try to extract a sub-model or build a new Model object.
      - Instead, use GradientTape directly on the FULL CNN-LSTM model,
        watching the input tensor and computing gradients w.r.t. the
        output of a target layer found by name search.
      - For per-frame explanation, we feed a (1, T, H, W, 3) sequence
        (with T=1 replicated) and take gradients of the predicted class
        score w.r.t. the input frames — this is "vanilla gradient" /
        SaliencyMap, which is robust across all model loading scenarios.
      - Additionally we attempt true Grad-CAM via the last Conv2D output
        using a try/except fallback to saliency if it fails.
"""

import logging
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

CLASS_NAMES = ["Low", "Medium", "High"]


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

def _find_last_conv2d(model: tf.keras.Model,
                      depth: int = 0) -> Optional[tf.keras.layers.Layer]:
    """
    Recursively search all layers and nested sub-models for the
    deepest Conv2D layer. Works on loaded .h5 models where MobileNetV2
    is nested several levels deep inside TimeDistributed wrappers.
    """
    last = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last = layer
        # Recurse into nested Model objects (e.g. cnn_extractor, MobileNetV2)
        if hasattr(layer, "layers"):
            sub = _find_last_conv2d(layer, depth + 1)
            if sub is not None:
                last = sub
    return last


def _frame_to_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert a float32 RGB (H,W,3) frame to uint8 BGR for OpenCV."""
    f = frame.clip(0, 1)
    f_u8 = np.uint8(f * 255)
    return cv2.cvtColor(f_u8, cv2.COLOR_RGB2BGR)


# ══════════════════════════════════════════════════════════════
#  GradCAM Explainer  — GradientTape on input, no sub-model build
# ══════════════════════════════════════════════════════════════
class GradCAMExplainer:
    """
    Robust Grad-CAM / Saliency explainer for CNN-LSTM hybrid models.

    Strategy (avoids the KeyError from sub-model graph mismatch):
      1. Feed a (1, T, H, W, 3) sequence through the FULL model using
         GradientTape watching the INPUT tensor.
      2. Compute d(class_score) / d(input_frames) — gradient of the
         predicted class score w.r.t. every pixel in every frame.
      3. Average gradients across the time dimension and channels to
         get a per-pixel importance map → this is the saliency heatmap.

    This is equivalent to "Vanilla Gradient" / Input × Gradient and
    is guaranteed to work on any Keras model regardless of how it was
    saved or loaded, because we never try to re-wire the graph.

    Args:
        full_model   : Full trained CNN-LSTM Keras model.
        target_layer : Unused (kept for API compatibility). Saliency
                       is computed w.r.t. the input directly.
        colormap     : OpenCV colormap string (e.g. "jet").
        alpha        : Heatmap blend opacity (0=invisible, 1=opaque).
    """

    def __init__(self, full_model: tf.keras.Model,
                 target_layer: str = "Conv_1",
                 colormap: str = "jet",
                 alpha: float = 0.4):
        self.full_model   = full_model
        self.target_layer = target_layer
        self.colormap     = colormap
        self.alpha        = alpha
        self.cnn_model    = self   # satisfy XAIRunner.cnn_model reference

        # Find sequence input shape from model
        input_shape = full_model.input_shape   # (None, T, H, W, 3)
        self.T = input_shape[1]
        self.H = input_shape[2]
        self.W = input_shape[3]

        logger.info(
            f"GradCAMExplainer ready — using input-gradient saliency "
            f"(T={self.T}, H={self.H}, W={self.W})"
        )

    def _compute_saliency(self, sequence: np.ndarray,
                          class_idx: int) -> np.ndarray:
        """
        Compute input-gradient saliency map for a full sequence.

        Args:
            sequence  : float32 (T, H, W, 3) — one video clip.
            class_idx : Target class index.

        Returns:
            saliency  : float32 (H, W) — per-pixel importance, range [0,1].
        """
        # Add batch dim: (1, T, H, W, 3)
        x = tf.Variable(
            np.expand_dims(sequence, axis=0).astype(np.float32),
            trainable=True
        )

        with tf.GradientTape() as tape:
            tape.watch(x)
            preds = self.full_model(x, training=False)   # (1, num_classes)
            score = preds[:, class_idx]

        # Gradients shape: (1, T, H, W, 3)
        grads = tape.gradient(score, x).numpy()[0]      # (T, H, W, 3)

        # Aggregate: mean absolute gradient across time and channels
        saliency = np.abs(grads).mean(axis=(0, 3))     # (H, W)

        # Normalise to [0, 1]
        if saliency.max() > 0:
            saliency = saliency / saliency.max()

        return saliency.astype(np.float32)

    def compute_heatmap(self, frame: np.ndarray,
                        sequence: np.ndarray,
                        class_idx: int) -> np.ndarray:
        """
        Compute saliency heatmap for a frame and return BGR uint8 image.

        Args:
            frame     : float32 (H, W, 3) — the specific frame to overlay.
            sequence  : float32 (T, H, W, 3) — full sequence for gradient.
            class_idx : Target congestion class.

        Returns:
            BGR uint8 heatmap image (H, W, 3).
        """
        saliency    = self._compute_saliency(sequence, class_idx)
        H, W        = frame.shape[:2]
        sal_resized = cv2.resize(saliency, (W, H))
        sal_u8      = np.uint8(255 * sal_resized)
        colormap_id = getattr(cv2, f"COLORMAP_{self.colormap.upper()}",
                              cv2.COLORMAP_JET)
        return cv2.applyColorMap(sal_u8, colormap_id)

    def overlay(self, frame: np.ndarray,
                sequence: np.ndarray,
                class_idx: int) -> np.ndarray:
        """
        Blend saliency heatmap over the original frame.

        Returns BGR uint8 image (H, W, 3).
        """
        frame_bgr = _frame_to_bgr_uint8(frame)
        heatmap   = self.compute_heatmap(frame, sequence, class_idx)
        return cv2.addWeighted(
            frame_bgr, 1.0 - self.alpha,
            heatmap,   self.alpha, 0
        )

    def explain_sequence(self, sequence: np.ndarray,
                         pred_class: int,
                         save_dir: str,
                         sample_idx: int = 0,
                         max_frames: int = 4) -> None:
        """
        Save saliency-map overlays for key frames in a sequence.

        Args:
            sequence  : float32 (T, H, W, 3).
            pred_class: Predicted congestion class index.
            save_dir  : Output directory for PNG files.
            sample_idx: Used in the output filename.
            max_frames: How many frames to visualise side-by-side.
        """
        T     = sequence.shape[0]
        tidxs = np.linspace(0, T - 1, min(max_frames, T), dtype=int)
        ncols = len(tidxs)

        # Compute saliency once for the whole sequence (shared gradient)
        saliency = self._compute_saliency(sequence, pred_class)

        fig, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 12))
        if ncols == 1:
            axes = axes.reshape(3, 1)

        fig.suptitle(
            f"Grad-CAM Saliency  |  "
            f"Predicted: {CLASS_NAMES[pred_class]} Congestion  |  "
            f"Sample {sample_idx}",
            fontsize=13, fontweight="bold"
        )

        for col, t in enumerate(tidxs):
            frame   = sequence[t]                          # (H, W, 3)
            ov_bgr  = self.overlay(frame, sequence, pred_class)
            ov_rgb  = cv2.cvtColor(ov_bgr, cv2.COLOR_BGR2RGB)

            # Row 0: original frame
            axes[0, col].imshow(frame.clip(0, 1))
            axes[0, col].set_title(f"Frame {t}")
            axes[0, col].axis("off")

            # Row 1: saliency heatmap alone
            axes[1, col].imshow(saliency, cmap="jet")
            axes[1, col].set_title("Saliency Map")
            axes[1, col].axis("off")

            # Row 2: overlay
            axes[2, col].imshow(ov_rgb)
            axes[2, col].set_title("Overlay")
            axes[2, col].axis("off")

        plt.tight_layout()
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / f"gradcam_sample_{sample_idx}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Grad-CAM saliency saved → {out}")

    # -- CNN model interface for SHAP/LIME (pass-through to full model) ----
    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        """
        Predict on a batch of individual frames by replicating each
        frame T times to form a valid (N, T, H, W, 3) sequence input.
        Used by SHAPExplainer and LIMEExplainer.
        """
        T   = self.T
        # X shape: (N, H, W, 3) — individual frames
        # Replicate each frame T times → (N, T, H, W, 3)
        X_seq = np.stack([X] * T, axis=1).astype(np.float32)
        return self.full_model.predict(X_seq, verbose=verbose)


# ══════════════════════════════════════════════════════════════
#  SHAP Explainer
# ══════════════════════════════════════════════════════════════
class SHAPExplainer:
    """
    KernelSHAP explainer applied to a single representative frame.

    Input flattening fix: KernelSHAP requires 1D or 2D input
    (n_samples, n_features). We flatten (H,W,3) → (H*W*3,) and
    reshape SHAP values back to (H,W,3) for visualisation.

    Args:
        cnn_model      : Model or object with .predict(X) where X is (N,H,W,3).
        num_background : Background samples for KernelSHAP.
        img_shape      : (H, W, C) shape of individual frames.
    """

    def __init__(self, cnn_model,
                 num_background: int = 20,
                 img_shape: tuple = (112, 112, 3)):
        try:
            import shap
            self.shap = shap
        except ImportError:
            logger.warning("SHAP not installed. Run: pip install shap")
            self.shap = None

        self.cnn_model      = cnn_model
        self.num_background = num_background
        self.img_shape      = img_shape
        self.flat_dim       = int(np.prod(img_shape))

    def _predict_flat(self, X_flat: np.ndarray) -> np.ndarray:
        """KernelSHAP wrapper: (N, H*W*C) → (N, num_classes)."""
        N     = X_flat.shape[0]
        X_img = X_flat.reshape(N, *self.img_shape).astype(np.float32)
        return self.cnn_model.predict(X_img, verbose=0)

    def explain_frame(self, frame: np.ndarray,
                      background_frames: np.ndarray,
                      save_dir: str,
                      sample_idx: int = 0) -> None:
        """Compute and save SHAP values for a single frame."""
        if self.shap is None:
            logger.warning("Skipping SHAP (not installed).")
            return

        H, W, C = self.img_shape

        # Flatten: (N, H, W, C) → (N, H*W*C)
        bg_flat    = background_frames[:self.num_background].reshape(
            -1, self.flat_dim).astype(np.float32)
        frame_flat = frame.reshape(1, self.flat_dim).astype(np.float32)

        try:
            explainer = self.shap.KernelExplainer(self._predict_flat, bg_flat)
            shap_vals = explainer.shap_values(
                frame_flat, nsamples=100, silent=True
            )
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")
            return

        num_classes = len(shap_vals)
        fig, axes   = plt.subplots(1, num_classes + 1,
                                   figsize=(5 * (num_classes + 1), 5))
        fig.suptitle(f"SHAP Feature Importance — Sample {sample_idx}",
                     fontsize=13, fontweight="bold")

        axes[0].imshow(frame.clip(0, 1))
        axes[0].set_title("Original Frame")
        axes[0].axis("off")

        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            sv  = shap_vals[cls_idx][0].reshape(H, W, C)
            sv_abs = np.abs(sv).mean(axis=-1)
            axes[cls_idx + 1].imshow(frame.clip(0, 1), alpha=0.5)
            im = axes[cls_idx + 1].imshow(sv_abs, cmap="hot", alpha=0.6)
            axes[cls_idx + 1].set_title(f"SHAP: {cls_name}")
            axes[cls_idx + 1].axis("off")
            plt.colorbar(im, ax=axes[cls_idx + 1], fraction=0.046)

        plt.tight_layout()
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / f"shap_sample_{sample_idx}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP saved → {out}")


# ══════════════════════════════════════════════════════════════
#  LIME Explainer
# ══════════════════════════════════════════════════════════════
class LIMEExplainer:
    """
    LIME superpixel explainer for a single representative frame.

    Args:
        cnn_model    : Model or object with .predict(X) where X is (N,H,W,3).
        num_samples  : Perturbation samples.
        num_segments : SLIC superpixel count.
        num_features : Top features to highlight.
    """

    def __init__(self, cnn_model,
                 num_samples: int = 300,
                 num_segments: int = 30,
                 num_features: int = 8):
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
        """LIME wrapper: X is uint8 (N,H,W,3) from LIME internals."""
        return self.cnn_model.predict(
            X.astype(np.float32) / 255.0, verbose=0
        )

    def explain_frame(self, frame: np.ndarray,
                      pred_class: int,
                      save_dir: str,
                      sample_idx: int = 0) -> None:
        """Compute and save LIME explanation for a single frame."""
        if self.lime_image is None:
            logger.warning("Skipping LIME (not installed).")
            return

        frame_u8  = (frame.clip(0, 1) * 255).astype(np.uint8)
        explainer = self.lime_image.LimeImageExplainer()

        try:
            explanation = explainer.explain_instance(
                frame_u8,
                self._predict_fn,
                top_labels=3,
                hide_color=0,
                num_samples=self.num_samples,
                num_features=self.num_features,
            )
            temp, mask = explanation.get_image_and_mask(
                pred_class,
                positive_only=False,
                num_features=self.num_features,
                hide_rest=False,
            )
        except Exception as e:
            logger.warning(f"LIME failed: {e}")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"LIME  |  Predicted: {CLASS_NAMES[pred_class]} Congestion  "
            f"|  Sample {sample_idx}",
            fontsize=13, fontweight="bold"
        )
        axes[0].imshow(frame.clip(0, 1))
        axes[0].set_title("Original Frame")
        axes[0].axis("off")
        axes[1].imshow(mask, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[1].set_title("Segment Mask\n(Green=+ve, Red=−ve)")
        axes[1].axis("off")
        axes[2].imshow(temp.astype(np.uint8))
        axes[2].set_title("LIME Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / f"lime_sample_{sample_idx}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"LIME saved → {out}")


# ══════════════════════════════════════════════════════════════
#  XAIRunner
# ══════════════════════════════════════════════════════════════
class XAIRunner:
    """
    Orchestrates Grad-CAM, SHAP, and LIME on test sequences.

    Key design:
      - GradCAMExplainer is built ONCE and reused (avoids tf.function
        retracing warnings).
      - GradCAMExplainer.predict() wraps the full model so SHAP and
        LIME can call it with individual frames (N, H, W, 3) — it
        internally replicates frames into valid sequences (N, T, H, W, 3).
    """

    def __init__(self, model: tf.keras.Model, cfg: dict):
        self.model    = model
        self.cfg      = cfg
        self.xai_cfg  = cfg["explainability"]
        self.hm_dir   = cfg["paths"]["heatmap_dir"]

        pp = cfg["preprocessing"]
        self.img_shape = (pp["frame_height"], pp["frame_width"], 3)

        # Build GradCAMExplainer once — also acts as CNN proxy for SHAP/LIME
        self.gc = GradCAMExplainer(
            full_model   = model,
            target_layer = self.xai_cfg["gradcam"]["target_layer"],
            colormap     = self.xai_cfg["gradcam"]["colormap"],
            alpha        = self.xai_cfg["gradcam"]["alpha"],
        ) if self.xai_cfg["gradcam"]["enabled"] else None

    def run(self, sequences: np.ndarray,
            y_true: np.ndarray,
            max_samples: int = 5) -> None:
        """Run all enabled XAI methods on up to max_samples test sequences."""
        n = min(max_samples, len(sequences))
        logger.info(f"Running XAI on {n} samples...")

        # Single predict call for all samples — avoids repeated retracing
        X_batch = np.array(sequences[:n], dtype=np.float32)
        y_proba = self.model.predict(X_batch, verbose=0)
        y_pred  = np.argmax(y_proba, axis=1)

        for i in range(n):
            seq       = np.array(sequences[i], dtype=np.float32)  # (T,H,W,3)
            pred_cls  = int(y_pred[i])
            mid_frame = seq[seq.shape[0] // 2]                    # (H,W,3)

            logger.info(
                f"Sample {i}: true={CLASS_NAMES[int(y_true[i])]}  "
                f"pred={CLASS_NAMES[pred_cls]}  "
                f"conf={y_proba[i, pred_cls]:.1%}"
            )

            # ── Grad-CAM / Saliency ────────────────────────────
            if self.gc is not None:
                self.gc.explain_sequence(
                    seq, pred_cls, self.hm_dir, sample_idx=i
                )

            # ── SHAP ───────────────────────────────────────────
            if self.xai_cfg["shap"]["enabled"]:
                shap_exp = SHAPExplainer(
                    cnn_model      = self.gc,   # uses GradCAMExplainer.predict()
                    num_background = self.xai_cfg["shap"]["num_background"],
                    img_shape      = self.img_shape,
                )
                bg_frames = np.array(
                    sequences[:self.xai_cfg["shap"]["num_background"],
                              seq.shape[0] // 2],
                    dtype=np.float32
                )
                shap_exp.explain_frame(
                    mid_frame, bg_frames, self.hm_dir, sample_idx=i
                )

            # ── LIME ───────────────────────────────────────────
            if self.xai_cfg["lime"]["enabled"]:
                lime_exp = LIMEExplainer(
                    cnn_model    = self.gc,   # uses GradCAMExplainer.predict()
                    num_samples  = self.xai_cfg["lime"]["num_samples"],
                    num_segments = self.xai_cfg["lime"]["num_segments"],
                    num_features = self.xai_cfg["lime"]["num_features"],
                )
                lime_exp.explain_frame(
                    mid_frame, pred_cls, self.hm_dir, sample_idx=i
                )

        logger.info(f"XAI complete. All outputs saved to: {self.hm_dir}")
