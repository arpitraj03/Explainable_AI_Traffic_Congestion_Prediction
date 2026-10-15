# Explainable AI-Based Traffic Congestion Prediction
## Hybrid Deep Learning (CNN + LSTM) with Grad-CAM / SHAP / LIME

---

## 📁 Project Structure

```
traffic_congestion_ai/
│
├── src/
│   ├── preprocessing.py        # Video frame extraction, resizing, normalization
│   ├── augmentation.py         # Data augmentation for frames
│   ├── dataset.py              # Dataset builder & train/test splitter
│   ├── model.py                # Hybrid CNN-LSTM architecture
│   ├── train.py                # Training loop with metrics
│   ├── evaluate.py             # Evaluation: accuracy, precision, recall, F1
│   ├── explainability.py       # Grad-CAM, SHAP, LIME implementations
│   ├── visualizer.py           # Heatmap overlays, congestion visualizations
│   └── predict.py              # Real-time inference on new CCTV videos
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard for live results
│
├── data/
│   ├── raw/                    # Place raw CCTV .mp4/.avi videos here
│   ├── processed/              # Extracted & preprocessed frames
│   └── augmented/              # Augmented frame sequences
│
├── models/
│   └── (saved .h5 / .pt files will appear here after training)
│
├── outputs/
│   ├── visualizations/         # Congestion prediction overlays
│   ├── heatmaps/               # Grad-CAM heatmap outputs
│   └── reports/                # Evaluation metrics & plots
│
├── tests/
│   └── test_pipeline.py        # Unit tests for all modules
│
├── notebooks/
│   └── exploration.ipynb       # EDA notebook
│
├── requirements.txt
├── config.yaml                 # All hyperparameters & paths
├── run_pipeline.py             # End-to-end runner
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your CCTV videos
Place your `.mp4` or `.avi` CCTV video files into the `data/raw/` directory.
Name them with their congestion label suffix for auto-labeling:
- `highway_low_001.mp4`   → label: **low**
- `intersection_medium_003.mp4` → label: **medium**
- `city_center_high_002.mp4`  → label: **high**

Small browser-friendly previews are available in [`data/previews/`](data/previews/). The original full-resolution videos remain in `data/raw/` and are stored with Git LFS.

### 3. Run the full pipeline
```bash
python run_pipeline.py --video_dir data/raw/ --epochs 30 --batch_size 8
```

### 4. Run inference on a new video
```bash
python src/predict.py --video path/to/new_cctv_video.mp4
```

### 5. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🧠 Model Architecture

```
Input Frames (T x H x W x C)
        │
    ┌───▼────────────────────────────┐
    │  CNN Block (MobileNetV2 base)  │  ← Spatial feature extraction per frame
    │  + GlobalAveragePooling        │
    └───────────────┬────────────────┘
                    │  Feature vectors [T x 1280]
    ┌───────────────▼────────────────┐
    │  LSTM Block (2 layers)         │  ← Temporal sequence modelling
    │  hidden=256, dropout=0.3       │
    └───────────────┬────────────────┘
                    │
    ┌───────────────▼────────────────┐
    │  Dense Head                    │
    │  FC(128) → FC(64) → FC(3)      │  ← 3 classes: Low / Medium / High
    └────────────────────────────────┘
```

---

## 📊 Congestion Classes

| Label  | Meaning                              |
|--------|--------------------------------------|
| 0 - Low    | Free-flowing traffic, <30% density   |
| 1 - Medium | Moderate congestion, 30–70% density  |
| 2 - High   | Heavy congestion / standstill, >70%  |

---

## 🔍 Explainability Methods

| Method   | What it shows                                                    |
|----------|------------------------------------------------------------------|
| Grad-CAM | Heatmap highlighting frame regions driving the prediction        |
| SHAP     | Feature importance scores for each spatial region               |
| LIME     | Superpixel importance map, perturbed local explanation           |

---

## 📈 Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (per class & macro-averaged)
- Confusion matrix
- Per-video congestion timeline plots

---

## ⚙️ Configuration (`config.yaml`)
All hyperparameters, paths, and model settings are centralised in `config.yaml`.
Edit it before running to customise the experiment.
