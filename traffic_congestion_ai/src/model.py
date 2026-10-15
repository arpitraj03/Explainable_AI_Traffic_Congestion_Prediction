"""
model.py
========
Hybrid CNN-LSTM Architecture for Traffic Congestion Prediction

Architecture overview:
  ┌─────────────────────────────────────────────────────────┐
  │  Input: (batch, T, H, W, 3)                              │
  │                                                          │
  │  TimeDistributed(CNN Backbone)  ← spatial per frame      │
  │    MobileNetV2 / ResNet50 / EfficientNetB0               │
  │    + GlobalAveragePooling2D                              │
  │                                                          │
  │  → (batch, T, feature_dim)                               │
  │                                                          │
  │  Stacked Bidirectional LSTM     ← temporal dynamics      │
  │    Layer 1: 256 units, return_sequences=True             │
  │    Layer 2: 128 units, return_sequences=False            │
  │                                                          │
  │  Dense Head                                              │
  │    Dense(128, relu) → Dropout → Dense(64, relu)          │
  │    → Dense(num_classes, softmax)                         │
  └─────────────────────────────────────────────────────────┘

Design choices:
  - TimeDistributed wrapper applies the same CNN to each frame
    independently, sharing weights across time steps.
  - Bidirectional LSTM captures both forward and backward temporal
    patterns (e.g. traffic building up and dissipating).
  - The CNN backbone is initialised with ImageNet weights and kept
    frozen by default to act as a feature extractor. Fine-tuning
    can be enabled by setting cnn_trainable: true in config.yaml.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import yaml
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  CNN Backbone factory
# ══════════════════════════════════════════════════════════════
def build_cnn_backbone(name: str, input_shape: tuple,
                       trainable: bool = False) -> tf.keras.Model:
    """
    Instantiate a pre-trained CNN backbone.

    The backbone is used as a feature extractor inside a
    TimeDistributed wrapper, so it must accept (H, W, 3) inputs
    and return a spatial feature map.

    Args:
        name        : 'mobilenetv2', 'resnet50', or 'efficientnetb0'.
        input_shape : (H, W, 3) tuple.
        trainable   : If False, all backbone weights are frozen.

    Returns:
        Keras model ending at the final conv feature map.
    """
    kwargs = dict(include_top=False, weights="imagenet", input_shape=input_shape)

    if name == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(**kwargs)
    elif name == "resnet50":
        base = tf.keras.applications.ResNet50(**kwargs)
    elif name == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(**kwargs)
    else:
        raise ValueError(f"Unknown CNN backbone: {name}. "
                         "Choose mobilenetv2 | resnet50 | efficientnetb0")

    base.trainable = trainable
    logger.info(f"CNN backbone: {name} | trainable={trainable} | "
                f"output shape: {base.output_shape}")
    return base


# ══════════════════════════════════════════════════════════════
#  CongestionModel
# ══════════════════════════════════════════════════════════════
class CongestionModel:
    """
    Builder class for the Hybrid CNN-LSTM congestion prediction model.

    Args:
        cfg : Full project config dict.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        mc  = cfg["model"]
        pp  = cfg["preprocessing"]

        self.T           = pp["sequence_length"]
        self.H           = pp["frame_height"]
        self.W           = pp["frame_width"]
        self.C           = pp["num_channels"]
        self.backbone    = mc["cnn_backbone"]
        self.trainable   = mc["cnn_trainable"]
        self.lstm_units  = mc["lstm_units"]
        self.lstm_drop   = mc["lstm_dropout"]
        self.lstm_rec    = mc["lstm_recurrent_dropout"]
        self.dense_units = mc["dense_units"]
        self.dense_drop  = mc["dense_dropout"]
        self.num_classes = mc["num_classes"]

    def build(self) -> Model:
        """
        Construct and return the compiled Keras model.

        Returns:
            Compiled tf.keras.Model
        """
        frame_shape = (self.H, self.W, self.C)

        # ── CNN feature extractor ──────────────────────────────
        cnn_base = build_cnn_backbone(
            self.backbone, frame_shape, trainable=self.trainable
        )

        # Wrap CNN in GlobalAveragePooling so each frame → 1-D vector
        cnn_input  = layers.Input(shape=frame_shape, name="cnn_input")
        cnn_feats  = cnn_base(cnn_input)
        cnn_feats  = layers.GlobalAveragePooling2D(name="gap")(cnn_feats)
        cnn_model  = Model(cnn_input, cnn_feats, name="cnn_extractor")

        # ── Sequence input ─────────────────────────────────────
        seq_input = layers.Input(
            shape=(self.T, self.H, self.W, self.C),
            name="sequence_input"
        )

        # Apply CNN to each frame → (batch, T, feature_dim)
        frame_features = layers.TimeDistributed(
            cnn_model, name="time_distributed_cnn"
        )(seq_input)

        # ── Stacked Bidirectional LSTM ─────────────────────────
        x = frame_features
        for i, units in enumerate(self.lstm_units):
            return_seq = (i < len(self.lstm_units) - 1)   # True for all but last
            x = layers.Bidirectional(
                layers.LSTM(
                    units,
                    return_sequences=return_seq,
                    dropout=self.lstm_drop,
                    recurrent_dropout=self.lstm_rec,
                    name=f"lstm_{i+1}"
                ),
                name=f"bilstm_{i+1}"
            )(x)
            x = layers.LayerNormalization(name=f"layer_norm_{i+1}")(x)

        # ── Dense classification head ──────────────────────────
        for j, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation="relu", name=f"dense_{j+1}")(x)
            x = layers.Dropout(self.dense_drop, name=f"dropout_{j+1}")(x)

        output = layers.Dense(
            self.num_classes, activation="softmax", name="predictions"
        )(x)

        model = Model(inputs=seq_input, outputs=output, name="CNN_LSTM_Congestion")

        # ── Compile ────────────────────────────────────────────
        self._compile(model)

        model.summary(print_fn=logger.info)
        return model

    def _compile(self, model: Model) -> None:
        """Compile with optimizer, loss, and metrics from config."""
        tc = self.cfg["training"]
        lr = tc["learning_rate"]

        if tc["optimizer"].lower() == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        elif tc["optimizer"].lower() == "sgd":
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(
            optimizer=opt,
            loss=tc["loss"],
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ]
        )

    @staticmethod
    def load(path: str, cfg: dict) -> Model:
        """Load a saved model from disk."""
        model = tf.keras.models.load_model(path)
        logger.info(f"Loaded model from {path}")
        return model


# ══════════════════════════════════════════════════════════════
#  Callbacks factory
# ══════════════════════════════════════════════════════════════
def build_callbacks(cfg: dict) -> List[tf.keras.callbacks.Callback]:
    """
    Build training callbacks:
      - ModelCheckpoint  : Save best weights
      - EarlyStopping    : Prevent overfitting
      - ReduceLROnPlateau: Reduce learning rate on plateau
      - TensorBoard      : Training curves
      - CSVLogger        : Save metrics to CSV
    """
    tc   = cfg["training"]
    paths = cfg["paths"]
    Path(paths["model_save_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["report_dir"]).mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=paths["model_checkpoint"],
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=tc["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=tc["reduce_lr_factor"],
            patience=tc["reduce_lr_patience"],
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="outputs/reports/tensorboard_logs",
            histogram_freq=1,
        ),
        tf.keras.callbacks.CSVLogger(
            f"{paths['report_dir']}/training_history.csv",
            append=False,
        ),
    ]
