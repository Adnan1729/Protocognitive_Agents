"""
Section V.A and V.C: feedforward ReLU classifier.

Architecture: Dense(128) -> Dropout -> Dense(64) -> Dropout
              -> Dense(32) -> Dense(16) -> Dense(1, sigmoid)
"""

from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score,
    precision_score, recall_score, roc_curve,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from . import config


def set_global_seed(seed: int = config.RANDOM_SEED) -> None:
    """Seed Python, numpy, and TensorFlow for deterministic runs.

    Note: TensorFlow on GPU has additional non-determinism sources that
    full determinism would require TF_DETERMINISTIC_OPS=1 to address.
    On CPU this seeding is sufficient.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_model(input_shape: int) -> Sequential:
    """Build the feedforward network used in the paper."""
    sizes = config.HIDDEN_LAYER_SIZES
    model = Sequential([
        Dense(sizes[0], activation="relu", input_shape=(input_shape,)),
        Dropout(config.DROPOUT_RATE),
        Dense(sizes[1], activation="relu"),
        Dropout(config.DROPOUT_RATE),
        Dense(sizes[2], activation="relu"),
        Dense(sizes[3], activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    return model


def train_model(model: Sequential, X_train, y_train):
    """Compile and fit the model with the paper's training schedule."""
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )
    lr_scheduler = ReduceLROnPlateau(
        factor=config.LR_REDUCE_FACTOR,
        patience=config.LR_REDUCE_PATIENCE,
    )

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1,
    )
    return history


def evaluate_model(model: Sequential, X_test, y_test) -> dict:
    """Compute predictions and the classification metrics in Table IV."""
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > config.CLASSIFICATION_THRESHOLD).astype(int).flatten()

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
        "confusion_matrix": cm,
    }