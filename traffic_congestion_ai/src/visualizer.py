"""
visualizer.py
=============
Congestion Visualisation Utilities

Produces:
  1. Per-frame congestion label overlays (text + colour band)
  2. Congestion heatmap overlays using density estimation
  3. Per-video congestion timeline plot
  4. Annotated output video (annotated MP4 / AVI)
"""

import sys
import logging
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# ── Congestion class colour map (BGR for OpenCV) ───────────────
CONGESTION_COLORS_BGR = {
    0: (34,  139, 34),    # Low    → Forest Green
    1: (0,   165, 255),   # Medium → Orange
    2: (0,   0,   220),   # High   → Red
}
CONGESTION_COLORS_RGB = {
    0: (34,  139, 34),    # Low
    1: (255, 165, 0),     # Medium
    2: (220, 0,   0),     # High
}
CLASS_NAMES = ["Low", "Medium", "High"]


# ══════════════════════════════════════════════════════════════
#  Frame Annotator
# ══════════════════════════════════════════════════════════════
class FrameAnnotator:
    """
    Annotates individual video frames with:
      - Congestion label + confidence score (top-left corner)
      - Colour-coded border strip (top bar)
      - Optional density heatmap overlay

    Args:
        frame_size : (width, height) of the output frame.
    """

    def __init__(self, frame_size: Tuple[int, int] = (224, 224)):
        self.W, self.H = frame_size

    def annotate(self, frame: np.ndarray,
                 pred_class: int,
                 confidence: float,
                 show_border: bool = True) -> np.ndarray:
        """
        Annotate a single frame.

        Args:
            frame       : uint8 RGB or float32 [0,1] (H, W, 3).
            pred_class  : Predicted congestion class index.
            confidence  : Softmax confidence for predicted class.
            show_border : Draw a colour band along the top edge.

        Returns:
            Annotated uint8 BGR frame.
        """
        # Ensure uint8 BGR
        if frame.dtype != np.uint8:
            frame = np.uint8(frame * 255)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out = frame_bgr.copy()

        color = CONGESTION_COLORS_BGR[pred_class]

        # ── Colour border strip (top 8 px) ─────────────────────
        if show_border:
            out[:8, :] = color

        # ── Background rectangle for text ──────────────────────
        label_text = f"{CLASS_NAMES[pred_class]:>6} ({confidence:.0%})"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,
                                       0.5, 1)
        cv2.rectangle(out, (5, 10), (5 + tw + 8, 10 + th + 8),
                      color, cv2.FILLED)

        # ── Text ───────────────────────────────────────────────
        cv2.putText(out, label_text,
                    (9, 10 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        return out

    def create_density_heatmap(self, frame: np.ndarray,
                               pred_class: int) -> np.ndarray:
        """
        Generate a synthetic density heatmap overlay based on brightness.

        For a real system this would use object detection or optical flow.
        Here we use frame brightness as a proxy for vehicle density.

        Returns:
            uint8 BGR overlay frame.
        """
        if frame.dtype != np.uint8:
            frame = np.uint8(frame * 255)

        gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Scale brightness by predicted congestion level
        scale   = 1.0 + pred_class * 0.5
        density = np.clip(gray.astype(float) * scale / 255.0, 0, 1)

        # Apply colour map
        density_u8  = np.uint8(density * 255)
        heatmap     = cv2.applyColorMap(density_u8, cv2.COLORMAP_JET)
        frame_bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        overlay     = cv2.addWeighted(frame_bgr, 0.6, heatmap, 0.4, 0)

        return overlay


# ══════════════════════════════════════════════════════════════
#  Timeline Plotter
# ══════════════════════════════════════════════════════════════
class TimelinePlotter:
    """
    Plots a congestion level timeline for an entire video.

    The x-axis represents time (frame sequence index), the y-axis
    shows the predicted congestion level (0=Low, 1=Medium, 2=High),
    and shaded bands indicate the congestion region.
    """

    @staticmethod
    def plot(predictions: List[int],
             confidences: List[float],
             video_name: str,
             save_dir: str,
             fps: float = 30.0,
             frame_skip: int = 2,
             sequence_length: int = 16) -> None:
        """
        Save a congestion timeline for a list of per-sequence predictions.

        Args:
            predictions    : List of class indices per sequence.
            confidences    : List of confidence scores per sequence.
            video_name     : Video filename (used in title and filename).
            save_dir       : Output directory.
            fps            : Video frames per second.
            frame_skip     : Frame skip used during extraction.
            sequence_length: Frames per sequence.
        """
        # Convert sequence indices to real-world seconds
        secs_per_seq = (sequence_length * frame_skip) / fps
        times = [i * secs_per_seq for i in range(len(predictions))]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                        gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(f"Traffic Congestion Timeline: {video_name}",
                     fontsize=14, fontweight="bold")

        # ── Top plot: congestion level ──────────────────────────
        colors = [CONGESTION_COLORS_RGB[p] for p in predictions]
        rgb_norm = [(r/255, g/255, b/255) for r, g, b in colors]

        # Area fill by congestion level
        for cls_idx, (level_name, fill_color) in enumerate(
            zip(CLASS_NAMES, ["#c8e6c9", "#ffe0b2", "#ffcdd2"])
        ):
            mask = [p == cls_idx for p in predictions]
            for i, (t, m) in enumerate(zip(times, mask)):
                if m:
                    ax1.axvspan(t, t + secs_per_seq, alpha=0.3,
                                color=fill_color)

        ax1.plot(times, predictions, color="#333333", linewidth=1.5, zorder=5)
        ax1.scatter(times, predictions, c=rgb_norm, s=40, zorder=6)
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(CLASS_NAMES)
        ax1.set_ylabel("Congestion Level")
        ax1.set_xlim(0, max(times) + secs_per_seq)
        ax1.grid(axis="x", alpha=0.3)

        # Legend
        patches = [
            mpatches.Patch(color=np.array(CONGESTION_COLORS_RGB[i]) / 255,
                           label=CLASS_NAMES[i])
            for i in range(3)
        ]
        ax1.legend(handles=patches, loc="upper right")

        # ── Bottom plot: confidence ─────────────────────────────
        ax2.fill_between(times, confidences, alpha=0.5, color="#2196F3")
        ax2.plot(times, confidences, color="#1565C0", linewidth=1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Confidence")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        out_path = Path(save_dir) / f"timeline_{Path(video_name).stem}.png"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Timeline saved → {out_path}")


# ══════════════════════════════════════════════════════════════
#  VideoWriter  (annotated output video)
# ══════════════════════════════════════════════════════════════
class AnnotatedVideoWriter:
    """
    Writes an annotated output video where each frame is overlaid
    with congestion level labels and heatmaps.

    Args:
        output_path : Path to output video file (.mp4).
        fps         : Output FPS.
        frame_size  : (width, height) of output frames.
    """

    def __init__(self, output_path: str, fps: float = 15.0,
                 frame_size: Tuple[int, int] = (224, 224)):
        self.output_path = output_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.annotator = FrameAnnotator(frame_size)
        self.fps = fps
        self.frame_size = frame_size

    def write_sequence(self, sequence: np.ndarray,
                       pred_class: int,
                       confidence: float,
                       include_heatmap: bool = True) -> None:
        """
        Write all frames of a sequence to the output video,
        each annotated with the predicted congestion level.
        """
        T = sequence.shape[0]
        W, H = self.frame_size

        for t in range(T):
            frame = sequence[t]   # (H, W, 3) float32

            if include_heatmap:
                # Blend heatmap into frame
                frame_hm = self.annotator.create_density_heatmap(frame, pred_class)
                frame_hm_rgb = cv2.cvtColor(frame_hm, cv2.COLOR_BGR2RGB)
                annotated = self.annotator.annotate(
                    frame_hm_rgb / 255.0, pred_class, confidence
                )
            else:
                annotated = self.annotator.annotate(frame, pred_class, confidence)

            resized = cv2.resize(annotated, (W, H))
            self.writer.write(resized)

    def release(self) -> None:
        """Finalise and close the video file."""
        self.writer.release()
        logger.info(f"Annotated video saved → {self.output_path}")
