# metrics.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# Emotion labels (MELD standard)
EMOTION_NAMES = [
    "neutral", "joy", "sadness",
    "anger", "fear", "disgust", "surprise"
]


def compute_metrics(y_true, y_pred):
    """
    Computes standard evaluation metrics.

    Returns
    -------
    dict with keys:
        accuracy
        macro_precision
        macro_recall
        uar
        macro_f1
    """

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "uar": recall,              # Explicit alias
        "macro_f1": f1
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    save_path,
    normalize=False
):
    """
    Plots and saves a confusion matrix.

    Parameters
    ----------
    y_true : list[int]
    y_pred : list[int]
    save_path : str
        Path to save the PNG
    normalize : bool
        If True, plots normalized confusion matrix
    """

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=EMOTION_NAMES,
        yticklabels=EMOTION_NAMES
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
