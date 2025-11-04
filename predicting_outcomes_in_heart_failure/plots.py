from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from predicting_outcomes_in_heart_failure.config import FIGURES_DIR


def save_confusion_matrix(y_true, y_pred, model_name, labels: list[str] | None = None) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig_path = FIGURES_DIR / f"{model_name}_confusion_matrix.png"

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    if labels is None:
        labels = ["0", "1"]
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.success(f"Saved confusion matrix → {fig_path}")
    return fig_path
