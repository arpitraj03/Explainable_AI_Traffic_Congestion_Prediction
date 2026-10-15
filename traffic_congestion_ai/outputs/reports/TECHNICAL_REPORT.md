# Technical Report
# Explainable AI-Based Traffic Congestion Prediction Using Hybrid Deep Learning

---

## 1. Executive Summary

This project implements a complete end-to-end pipeline for predicting traffic congestion levels from CCTV video data using a hybrid **CNN-LSTM** deep learning model, augmented with three complementary **Explainable AI (XAI)** techniques: Grad-CAM, SHAP, and LIME. The system is designed to operate in real-time on new CCTV inputs and produces both congestion-level predictions and human-interpretable visual explanations.

---

## 2. Problem Statement

Urban traffic management systems collect vast amounts of CCTV footage but lack automated tools to:
1. Quantify congestion in real time (beyond simple vehicle counts)
2. Explain *why* a prediction was made (critical for trust in safety-critical deployments)

This project addresses both challenges with a production-ready pipeline.

**Congestion Classes:**

| Class  | Description                         | Density   |
|--------|-------------------------------------|-----------|
| Low    | Free-flowing traffic                | < 30%     |
| Medium | Moderate slowdown                   | 30–70%    |
| High   | Heavy congestion / standstill       | > 70%     |

---

## 3. Dataset Handling

### 3.1 Input Format
- **Source**: CCTV video files (.mp4, .avi, .mov)
- **Label encoding**: Inferred from filename (e.g., `highway_high_003.mp4`)
- **No manual annotation needed** for files following the naming convention

### 3.2 Preprocessing Pipeline (`src/preprocessing.py`)

```
Raw video
    ↓
Frame sampling (every N frames, configurable via frame_skip)
    ↓
BGR → RGB conversion
    ↓
MOG2 background subtraction (isolates moving vehicles)
    ↓
Resize to 224×224 pixels (INTER_AREA)
    ↓
Normalise to [0, 1]
    ↓
Group into sequences of length T=16
    ↓
Save as (N, T, H, W, 3) NumPy arrays
```

**Background Subtraction**: MOG2 (Mixture of Gaussians v2) models the road background and isolates foreground vehicles. This reduces the model's sensitivity to lighting changes and static road features that are irrelevant to congestion prediction.

### 3.3 Dataset Splitting
- Stratified 70% / 15% / 15% train / val / test split
- Stratification ensures class balance in each split
- Reproducible via fixed `random_seed=42`

### 3.4 Data Augmentation (`src/augmentation.py`)
Temporal consistency is maintained: random parameters are sampled **once per sequence** and applied identically to all frames.

| Augmentation         | Details                             |
|----------------------|-------------------------------------|
| Horizontal flip      | 50% probability                     |
| Brightness           | ×[0.8, 1.2]                        |
| Contrast             | ×[0.8, 1.2]                        |
| Rotation             | ±10°                                |
| Zoom/crop            | Up to 10% random zoom               |
| Gaussian noise       | σ=0.02                              |

Offline augmentation generates 2× copies → 3× total dataset size.

---

## 4. Model Architecture (`src/model.py`)

### 4.1 Design Rationale

Traffic congestion prediction requires understanding **both spatial and temporal** patterns:
- **Spatial**: Vehicle density, road occupancy, queue formation at a given instant
- **Temporal**: Congestion buildup/dissipation over time; short-term trends

A pure CNN treats each frame independently. A pure LSTM cannot extract spatial features. The **hybrid CNN-LSTM** architecture captures both.

### 4.2 Architecture Details

```
Input: (batch, T=16, H=224, W=224, C=3)
    │
    ╠══ TimeDistributed(MobileNetV2 backbone) ══╗
    ║   - ImageNet pretrained weights           ║   Shared CNN weights
    ║   - Frozen (transfer learning)            ║   applied per frame
    ║   - Output: (batch, T, 7, 7, 1280)        ║
    ║   - GlobalAveragePooling2D                ║
    ║   - Output: (batch, T, 1280)              ║
    ╚═══════════════════════════════════════════╝
    │
    ╠══ Bidirectional LSTM (256 units) ═════════╗
    ║   - return_sequences=True                 ║   Captures forward &
    ║   - dropout=0.3, recurrent_dropout=0.2    ║   backward temporal
    ║   - LayerNormalization                    ║   dependencies
    ╠══ Bidirectional LSTM (128 units) ═════════╣
    ║   - return_sequences=False                ║
    ║   - LayerNormalization                    ║
    ╚═══════════════════════════════════════════╝
    │
    ╠══ Dense(128, ReLU) → Dropout(0.4) ════════╗
    ╠══ Dense(64,  ReLU) → Dropout(0.4) ════════╣   Classification head
    ╠══ Dense(3,   Softmax) ════════════════════╝
    │
    Output: (batch, 3) — Low / Medium / High probabilities
```

### 4.3 Key Design Choices

| Choice                   | Reason                                                        |
|--------------------------|---------------------------------------------------------------|
| MobileNetV2 backbone     | Efficient (3.4M params), excellent ImageNet features         |
| Frozen backbone          | Reduces overfitting on small traffic datasets                 |
| Bidirectional LSTM       | Captures both build-up and dissipation patterns               |
| LayerNormalization       | Stabilises LSTM training, reduces sensitivity to LR           |
| Class weights            | Handles imbalanced congestion class distribution              |
| TimeDistributed wrapper  | Shares CNN weights across frames → fewer parameters           |

### 4.4 Training Configuration

| Hyperparameter           | Value                |
|--------------------------|----------------------|
| Optimiser                | Adam                 |
| Learning rate            | 0.001                |
| LR scheduler             | Cosine decay         |
| Batch size               | 8                    |
| Max epochs               | 30                   |
| Early stopping patience  | 7                    |
| Loss function            | Categorical cross-entropy |

---

## 5. Explainability Techniques (`src/explainability.py`)

### 5.1 Grad-CAM (Gradient-weighted Class Activation Mapping)

**What it does**: Computes the gradient of the predicted class score with respect to the last convolutional feature map. Large gradients indicate that a feature map location strongly influenced the prediction.

**Mathematical formulation**:
```
α_k^c = (1/Z) Σ_i Σ_j  ∂y^c / ∂A^k_{ij}    (global average pooling of gradients)
L^c_Grad-CAM = ReLU(Σ_k  α_k^c · A^k)         (weighted combination of feature maps)
```

**Interpretation**: Hot (red/yellow) regions in the Grad-CAM overlay indicate where the CNN looked to make its congestion prediction. Dense vehicle regions should appear hot for "High" predictions.

**Implementation detail**: We apply Grad-CAM to the CNN sub-model extracted from the TimeDistributed wrapper, targeting the last convolutional block of MobileNetV2.

### 5.2 SHAP (SHapley Additive exPlanations)

**What it does**: Assigns each input feature (image region) a Shapley value — the average marginal contribution of that feature across all possible coalitions of features.

**Approach**: KernelSHAP treats the model as a black box. Image regions are masked and the change in prediction score is measured. Background samples are used as the reference distribution.

**Interpretation**: Regions with high positive SHAP values increased the predicted congestion probability; negative values decreased it.

### 5.3 LIME (Local Interpretable Model-Agnostic Explanations)

**What it does**: Perturbs the input image by turning superpixels on/off, generating a locally linear surrogate model around the specific prediction.

**Approach**: SLIC segmentation creates superpixels; the surrogate model assigns each a positive or negative importance score for the predicted class.

**Interpretation**: Green segments contributed positively to the congestion prediction; red segments had negative influence.

### 5.4 Comparison

| Method   | Scope    | Model access | Speed   | Spatial resolution |
|----------|----------|--------------|---------|-------------------|
| Grad-CAM | Local    | White-box    | Fast    | Medium (upsampled CNN maps) |
| SHAP     | Local    | Black-box    | Slow    | High (pixel-level) |
| LIME     | Local    | Black-box    | Medium  | Superpixel-level  |

---

## 6. Evaluation Metrics

The model is evaluated on the held-out test set using:

| Metric             | Formula                                  |
|--------------------|------------------------------------------|
| Accuracy           | (TP + TN) / Total                        |
| Precision (per class) | TP / (TP + FP)                        |
| Recall (per class)    | TP / (TP + FN)                        |
| F1-Score (per class)  | 2·P·R / (P + R)                       |
| Macro F1            | Mean of per-class F1 scores             |

Outputs: classification report, confusion matrix, per-class bar chart, confidence box plot.

---

## 7. System Requirements

- Python 3.9+
- TensorFlow 2.12+
- OpenCV 4.8+
- 8GB RAM (16GB recommended for large video datasets)
- GPU optional but recommended for training (CUDA 11.x)

---

## 8. Limitations and Future Work

| Limitation                              | Suggested Improvement                         |
|-----------------------------------------|-----------------------------------------------|
| Label from filename (manual effort)    | Auto-labelling via optical flow density       |
| CNN backbone frozen                     | Fine-tune after initial training convergence  |
| Single-camera analysis                  | Multi-camera fusion for network-level insight |
| No real-time streaming                  | Integrate with RTSP stream reader             |
| Synthetic demo data when no videos      | Release labelled benchmark CCTV dataset       |

---

## 9. File Structure Summary

```
traffic_congestion_ai/
├── src/
│   ├── preprocessing.py    ← Video → frame sequence extraction
│   ├── augmentation.py     ← Temporally consistent augmentation
│   ├── dataset.py          ← tf.data pipeline, splitting
│   ├── model.py            ← CNN-LSTM architecture
│   ├── train.py            ← Training loop
│   ├── evaluate.py         ← Metrics, confusion matrix
│   ├── explainability.py   ← Grad-CAM, SHAP, LIME
│   ├── visualizer.py       ← Frame annotation, timelines, video writing
│   └── predict.py          ← Inference on new videos
├── dashboard/app.py        ← Streamlit web dashboard
├── run_pipeline.py         ← End-to-end orchestration
├── config.yaml             ← All hyperparameters
├── requirements.txt
└── tests/test_pipeline.py  ← Unit tests
```
