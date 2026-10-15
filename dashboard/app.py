"""
dashboard/app.py
================
Streamlit Dashboard for Traffic Congestion AI

Provides a live web interface for:
  • Video upload and real-time congestion prediction
  • Grad-CAM heatmap visualisation per frame
  • Congestion timeline chart
  • Model evaluation metrics display
  • XAI explanation viewer

Run with: streamlit run dashboard/app.py
"""

import sys
import os
import yaml
import json
import tempfile
import logging
import numpy as np
from pathlib import Path
from collections import Counter

# Ensure src modules are importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import cv2

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Congestion AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-left: 5px solid;
    margin-bottom: 1rem;
}

.level-low    { border-color: #4CAF50; }
.level-medium { border-color: #FF9800; }
.level-high   { border-color: #F44336; }

.prediction-badge {
    display: inline-block;
    padding: 0.4em 1.2em;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.1em;
    letter-spacing: 0.05em;
}

.badge-low    { background: #e8f5e9; color: #2e7d32; }
.badge-medium { background: #fff3e0; color: #e65100; }
.badge-high   { background: #ffebee; color: #b71c1c; }

.info-box {
    background: #e3f2fd;
    border-left: 4px solid #1976d2;
    padding: 0.8rem 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}

.code-block {
    background: #1e1e1e;
    color: #d4d4d4;
    padding: 1rem;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────
CLASS_NAMES   = ["Low",    "Medium",  "High"]
CLASS_COLORS  = ["#4CAF50", "#FF9800", "#F44336"]
CLASS_ICONS   = ["🟢",     "🟡",      "🔴"]
CLASS_BADGES  = ["badge-low", "badge-medium", "badge-high"]
CLASS_CARDS   = ["level-low", "level-medium", "level-high"]


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_config():
    cfg_path = ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


@st.cache_resource
def load_model(model_path: str):
    """Load and cache the Keras model."""
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        return None


def extract_frames_from_upload(video_bytes: bytes, cfg: dict) -> list:
    """Save uploaded bytes to a temp file and extract sequences."""
    from preprocessing import FrameExtractor

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    extractor = FrameExtractor(
        height=cfg["preprocessing"]["frame_height"],
        width=cfg["preprocessing"]["frame_width"],
        sequence_length=cfg["preprocessing"]["sequence_length"],
        frame_skip=cfg["preprocessing"]["frame_skip"],
        use_bg_sub=cfg["preprocessing"]["background_subtraction"],
        normalize=cfg["preprocessing"]["normalize"],
    )

    sequences = extractor.extract_sequences(tmp_path)
    fps = extractor.get_video_fps(tmp_path)
    os.unlink(tmp_path)
    return sequences, fps


def make_timeline_chart(predictions, confidences, secs_per_seq):
    """Build a Plotly congestion timeline figure."""
    times  = [i * secs_per_seq for i in range(len(predictions))]
    colors = [CLASS_COLORS[p] for p in predictions]
    labels = [CLASS_NAMES[p]  for p in predictions]

    fig = go.Figure()

    # Confidence fill
    fig.add_trace(go.Scatter(
        x=times, y=confidences,
        fill="tozeroy",
        fillcolor="rgba(33, 150, 243, 0.15)",
        line=dict(color="rgba(33, 150, 243, 0.6)", width=1.5),
        name="Confidence",
        yaxis="y2",
    ))

    # Congestion level line
    fig.add_trace(go.Scatter(
        x=times, y=predictions,
        mode="lines+markers",
        line=dict(color="#333", width=2),
        marker=dict(color=colors, size=10, line=dict(width=2, color="#fff")),
        text=[f"{l}<br>{c:.0%}" for l, c in zip(labels, confidences)],
        hovertemplate="%{text}<br>Time: %{x:.1f}s<extra></extra>",
        name="Congestion Level",
    ))

    fig.update_layout(
        title="Congestion Timeline",
        xaxis_title="Time (seconds)",
        yaxis=dict(
            tickvals=[0, 1, 2],
            ticktext=CLASS_NAMES,
            title="Level",
            range=[-0.3, 2.3],
        ),
        yaxis2=dict(
            title="Confidence",
            overlaying="y",
            side="right",
            range=[0, 1.2],
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=360,
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        margin=dict(t=50, b=40, l=60, r=60),
    )
    return fig


def gradcam_for_frame(model, frame):
    """
    Compute a simple Grad-CAM heatmap for display in the dashboard.
    Returns an RGB PIL Image.
    """
    try:
        from explainability import GradCAMExplainer
        cfg = load_config()
        gc  = GradCAMExplainer(
            full_model=model,
            target_layer=cfg["explainability"]["gradcam"]["target_layer"],
            alpha=0.45,
        )
        overlay_bgr = gc.overlay(frame)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(overlay_rgb)
    except Exception as e:
        # Fallback: return original frame as PIL Image
        frame_uint8 = (frame * 255).astype(np.uint8)
        return Image.fromarray(frame_uint8)


# ══════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        cfg = load_config()
        model_path = cfg.get("paths", {}).get("model_checkpoint",
                                               "models/cnn_lstm_best.h5")
        st.markdown(f"**Model**: `{Path(model_path).name}`")
        model_exists = Path(ROOT / model_path).exists()
        if model_exists:
            st.success("✓ Model loaded")
        else:
            st.warning("⚠️ No trained model found.\nRun the pipeline first.")

        st.markdown("---")
        show_gradcam = st.checkbox("Show Grad-CAM Heatmaps", value=True)
        show_confidence = st.checkbox("Show Confidence Bars", value=True)
        max_frames_display = st.slider("Frames to display", 2, 8, 4)

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "Explainable AI traffic congestion prediction using a "
            "**CNN-LSTM hybrid** model with Grad-CAM, SHAP, and LIME explanations."
        )

    return model_path, show_gradcam, show_confidence, max_frames_display


# ══════════════════════════════════════════════════════════════
#  Tab: Live Prediction
# ══════════════════════════════════════════════════════════════
def tab_predict(model, cfg, show_gradcam, show_confidence, max_frames):
    st.markdown("### 📹 Upload a CCTV Video for Prediction")
    st.markdown(
        '<div class="info-box">Supported formats: .mp4, .avi, .mov &nbsp;|&nbsp; '
        'Frames are automatically extracted and fed to the CNN-LSTM model.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Choose video file", type=["mp4", "avi", "mov"],
        label_visibility="collapsed"
    )

    if uploaded is None:
        # Demo mode with synthetic data
        st.markdown("---")
        st.markdown("#### 🧪 Demo Mode (no video uploaded)")
        if st.button("▶ Run Demo with Synthetic Sequences"):
            _run_demo(model, cfg, show_gradcam, show_confidence, max_frames)
        return

    # ── Process uploaded video ─────────────────────────────────
    with st.spinner("Extracting frames..."):
        video_bytes = uploaded.read()
        try:
            sequences, fps = extract_frames_from_upload(video_bytes, cfg)
        except Exception as e:
            st.error(f"Error processing video: {e}")
            return

    if not sequences:
        st.error("Could not extract any sequences from the video.")
        return

    st.success(f"Extracted **{len(sequences)} sequences** from `{uploaded.name}`")

    # ── Predict ────────────────────────────────────────────────
    if model is None:
        st.warning("No trained model loaded — showing mock predictions.")
        rng = np.random.default_rng(42)
        X = np.stack(sequences[:10], axis=0)
        predictions  = rng.integers(0, 3, size=len(X)).tolist()
        confidences  = rng.uniform(0.55, 0.99, size=len(X)).tolist()
    else:
        with st.spinner("Running inference..."):
            X        = np.stack(sequences, axis=0)
            y_proba  = model.predict(X, batch_size=4, verbose=0)
            predictions  = np.argmax(y_proba, axis=1).tolist()
            confidences  = y_proba.max(axis=1).tolist()

    # ── Overall summary ────────────────────────────────────────
    counts   = Counter(predictions)
    dominant = max(counts, key=counts.get)
    total    = len(predictions)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card {CLASS_CARDS[dominant]}">'
            f'<div style="font-size:0.8em;color:#666">Overall Assessment</div>'
            f'<span class="prediction-badge {CLASS_BADGES[dominant]}">'
            f'{CLASS_ICONS[dominant]} {CLASS_NAMES[dominant]} CONGESTION</span>'
            f'</div>', unsafe_allow_html=True
        )
    for cls_idx, col in zip([0, 1, 2], [c2, c3, c4]):
        pct = counts.get(cls_idx, 0) / total * 100
        with col:
            st.markdown(
                f'<div class="metric-card {CLASS_CARDS[cls_idx]}">'
                f'<div style="font-size:0.8em;color:#666">'
                f'{CLASS_ICONS[cls_idx]} {CLASS_NAMES[cls_idx]}</div>'
                f'<div style="font-size:1.5em;font-weight:700">'
                f'{pct:.0f}%</div>'
                f'<div style="font-size:0.75em;color:#888">'
                f'{counts.get(cls_idx,0)} / {total} seq</div>'
                f'</div>', unsafe_allow_html=True
            )

    # ── Timeline chart ─────────────────────────────────────────
    secs_per_seq = (
        cfg["preprocessing"]["sequence_length"] *
        cfg["preprocessing"]["frame_skip"]
    ) / max(fps, 1)

    fig = make_timeline_chart(predictions, confidences, secs_per_seq)
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-frame visualisation ────────────────────────────────
    st.markdown("### 🖼 Frame Visualisation")
    n_display = min(max_frames, len(sequences))
    frame_cols = st.columns(n_display)

    for col_idx, seq_idx in enumerate(
        np.linspace(0, len(sequences) - 1, n_display, dtype=int)
    ):
        with frame_cols[col_idx]:
            seq   = sequences[seq_idx]
            pred  = predictions[seq_idx]
            conf  = confidences[seq_idx]
            frame = seq[seq.shape[0] // 2]   # middle frame

            if show_gradcam and model is not None:
                img = gradcam_for_frame(model, frame)
            else:
                img = Image.fromarray((frame * 255).astype(np.uint8))

            st.image(img, caption=f"Seq {seq_idx}", use_container_width=True)
            st.markdown(
                f'<div style="text-align:center">'
                f'<span class="prediction-badge {CLASS_BADGES[pred]}">'
                f'{CLASS_ICONS[pred]} {CLASS_NAMES[pred]}</span></div>',
                unsafe_allow_html=True
            )
            if show_confidence:
                st.progress(float(conf), text=f"{conf:.0%}")


def _run_demo(model, cfg, show_gradcam, show_confidence, max_frames):
    """Run demo with synthetic data."""
    rng = np.random.default_rng(99)
    T = cfg["preprocessing"]["sequence_length"]
    H = cfg["preprocessing"]["frame_height"]
    W = cfg["preprocessing"]["frame_width"]

    sequences = []
    # Simulate increasing congestion scenario
    for cls_idx, n in [(0, 5), (1, 8), (2, 7)]:
        brightness = 0.2 + cls_idx * 0.3
        for _ in range(n):
            seq = rng.normal(brightness, 0.05, (T, H, W, 3)).clip(0, 1).astype(np.float32)
            sequences.append(seq)

    rng.shuffle(sequences)
    preds = [0]*5 + [1]*8 + [2]*7
    confs = rng.uniform(0.6, 0.95, len(preds)).tolist()

    fig = make_timeline_chart(preds, confs, 1.5)
    st.plotly_chart(fig, use_container_width=True)

    n_display = min(max_frames, len(sequences))
    cols = st.columns(n_display)
    for i, col in enumerate(cols):
        seq   = sequences[i]
        pred  = preds[i]
        frame = seq[T // 2]
        img   = Image.fromarray((frame * 255).astype(np.uint8))
        with col:
            st.image(img, caption=f"Seq {i}", use_container_width=True)
            st.markdown(
                f'<div style="text-align:center">'
                f'<span class="prediction-badge {CLASS_BADGES[pred]}">'
                f'{CLASS_ICONS[pred]} {CLASS_NAMES[pred]}</span></div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════
#  Tab: Evaluation Metrics
# ══════════════════════════════════════════════════════════════
def tab_metrics(cfg):
    st.markdown("### 📊 Model Evaluation Metrics")

    report_dir = Path(ROOT / cfg["paths"]["report_dir"])

    # Try to load saved metrics
    report_file = report_dir / "classification_report.txt"
    if report_file.exists():
        with open(report_file) as f:
            st.code(f.read(), language=None)
    else:
        st.info("No evaluation report found. Run the pipeline to generate metrics.")
        # Show mock metrics for demonstration
        mock_metrics = {
            "Low":    {"Precision": 0.88, "Recall": 0.91, "F1": 0.89},
            "Medium": {"Precision": 0.79, "Recall": 0.82, "F1": 0.80},
            "High":   {"Precision": 0.93, "Recall": 0.87, "F1": 0.90},
        }
        fig = go.Figure()
        x = list(mock_metrics.keys())
        for metric in ["Precision", "Recall", "F1"]:
            y = [mock_metrics[cls][metric] for cls in x]
            fig.add_trace(go.Bar(name=metric, x=x, y=y))
        fig.update_layout(
            title="Per-Class Metrics (demo)",
            barmode="group",
            yaxis=dict(range=[0, 1.1]),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Show saved plots
    for img_file in ["confusion_matrix.png", "per_class_metrics.png",
                     "confidence_boxplot.png", "training_history.png"]:
        img_path = report_dir / img_file
        if img_path.exists():
            st.image(str(img_path), caption=img_file.replace("_", " ").title(),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  Tab: XAI Explanations
# ══════════════════════════════════════════════════════════════
def tab_xai(cfg):
    st.markdown("### 🔍 Explainability Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Grad-CAM")
        st.markdown(
            "Highlights which **spatial regions** in the frame most activated "
            "the CNN feature maps for the predicted congestion class."
        )
        hm_dir = Path(ROOT / cfg["paths"]["heatmap_dir"])
        gradcam_files = sorted(hm_dir.glob("gradcam_*.png"))
        for f in gradcam_files[:3]:
            st.image(str(f), caption=f.name, use_container_width=True)
        if not gradcam_files:
            st.info("Run the pipeline to generate Grad-CAM heatmaps.")

    with col2:
        st.markdown("#### SHAP")
        st.markdown(
            "Shapley values quantify each pixel region's **contribution** "
            "to the congestion prediction using game-theory-based attribution."
        )
        shap_files = sorted(hm_dir.glob("shap_*.png"))
        for f in shap_files[:3]:
            st.image(str(f), caption=f.name, use_container_width=True)
        if not shap_files:
            st.info("Run the pipeline to generate SHAP explanations.")

    with col3:
        st.markdown("#### LIME")
        st.markdown(
            "Identifies **superpixel segments** that positively or negatively "
            "influence the prediction via local linear surrogate modelling."
        )
        lime_files = sorted(hm_dir.glob("lime_*.png"))
        for f in lime_files[:3]:
            st.image(str(f), caption=f.name, use_container_width=True)
        if not lime_files:
            st.info("Run the pipeline to generate LIME explanations.")


# ══════════════════════════════════════════════════════════════
#  Tab: Architecture & Docs
# ══════════════════════════════════════════════════════════════
def tab_docs():
    st.markdown("### 🧠 Model Architecture")
    st.markdown("""
```
Input Frames: (batch × T × H × W × 3)
        │
┌───────▼─────────────────────────────────────┐
│  TimeDistributed(MobileNetV2)               │  ← Per-frame spatial features
│  + GlobalAveragePooling2D                   │    (ImageNet pretrained, frozen)
└───────────────┬─────────────────────────────┘
                │  (batch × T × 1280)
┌───────────────▼─────────────────────────────┐
│  Bidirectional LSTM (256 units)             │  ← Temporal forward+backward
│  LayerNorm + Bidirectional LSTM (128 units) │
└───────────────┬─────────────────────────────┘
                │
┌───────────────▼─────────────────────────────┐
│  Dense(128, ReLU) → Dropout(0.4)            │
│  Dense(64,  ReLU) → Dropout(0.4)            │
│  Dense(3,  Softmax)                         │  → Low / Medium / High
└─────────────────────────────────────────────┘
```
""")

    st.markdown("### 🚀 How to Run")
    st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add CCTV videos to data/raw/
#    Name format: <prefix>_<low|medium|high>_<id>.mp4

# 3. Run the full pipeline
python run_pipeline.py --epochs 30 --batch_size 8

# 4. Predict on a new video
python src/predict.py --video path/to/cctv.mp4

# 5. Launch this dashboard
streamlit run dashboard/app.py
""", language="bash")

    st.markdown("### 📦 Key Libraries")
    libs = {
        "TensorFlow 2.12+": "CNN-LSTM model, training",
        "OpenCV":           "Video I/O, frame processing",
        "SHAP":             "KernelSHAP explanations",
        "LIME":             "Superpixel explanations",
        "Streamlit":        "This dashboard",
        "scikit-learn":     "Metrics, data splitting",
        "Plotly":           "Interactive charts",
    }
    for lib, use in libs.items():
        st.markdown(f"- **{lib}**: {use}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    # ── Header ─────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;font-size:2em; line-height:1.3;">
🚦 Explainable AI-Based Traffic Congestion Prediction<br>
<span style="margin-left:38px;">Using Hybrid Deep Learning</span>
</h1>
        <p style="margin:0.5em 0 0 0;opacity:0.8;font-size:1.05em">
            Explainable Deep Learning · CNN-LSTM · Grad-CAM · SHAP · LIME
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────
    model_path, show_gradcam, show_confidence, max_frames = sidebar()
    cfg = load_config()

    if not cfg:
        st.error("config.yaml not found. Make sure you're running from the project root.")
        st.stop()

    # ── Load model ──────────────────────────────────────────────
    full_model_path = str(ROOT / model_path)
    model = load_model(full_model_path)

    # ── Tabs ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📹 Live Prediction",
        "📊 Evaluation Metrics",
        "🔍 XAI Explanations",
        "📖 Architecture & Docs",
    ])

    with tab1:
        tab_predict(model, cfg, show_gradcam, show_confidence, max_frames)
    with tab2:
        tab_metrics(cfg)  
    with tab3:
        tab_xai(cfg)
    with tab4:
        tab_docs()


if __name__ == "__main__":
    main()
