"""Static matplotlib artifacts for ROC, PR, confusion, and cost curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_COLORS = {
    "Logistic Regression": "#4F46E5",
    "Decision Tree": "#0EA5E9",
    "Random Forest": "#10B981",
    "XGBoost Classifier": "#EF4444",
    "Support Vector Machine": "#F59E0B",
}


def save_comparison_plots(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Write ROC, PR, and champion confusion/cost charts.

    Args:
        results: Per-model payloads produced by the training loop.
        output_dir: Directory that receives the PNG files.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_roc(results, output_dir / "roc_curves.png")
    _save_pr(results, output_dir / "pr_curves.png")
    champion = next((item for item in results if item.get("is_champion")), results[0])
    _save_confusion(champion, output_dir / "confusion_champion.png")
    _save_cost(champion, output_dir / "cost_curve.png")


def _save_roc(results: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    for item in results:
        color = MODEL_COLORS.get(item["model_name"], "#334155")
        axis.plot(
            item["roc_curve"]["fpr"],
            item["roc_curve"]["tpr"],
            color=color,
            label=f"{item['model_name']} (AUC {item['roc_auc']:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC curves")
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _save_pr(results: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    for item in results:
        color = MODEL_COLORS.get(item["model_name"], "#334155")
        axis.plot(
            item["pr_curve"]["recall_curve"],
            item["pr_curve"]["precision_curve"],
            color=color,
            label=f"{item['model_name']} (AP {item['pr_auc']:.3f})",
        )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-recall curves")
    axis.legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _save_confusion(champion: dict[str, Any], path: Path) -> None:
    matrix = champion["confusion_matrix"]
    values = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]], dtype=float)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks([0, 1], ["Pred Low", "Pred High"])
    axis.set_yticks([0, 1], ["Actual Low", "Actual High"])
    for row in range(2):
        for col in range(2):
            axis.text(col, row, int(values[row, col]), ha="center", va="center", color="#0F172A")
    axis.set_title(f"{champion['model_name']} confusion matrix")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _save_cost(champion: dict[str, Any], path: Path) -> None:
    curve = champion.get("cost_curve", [])
    figure, axis = plt.subplots(figsize=(7, 4))
    if curve:
        axis.plot(
            [point["threshold"] for point in curve],
            [point["expected_cost"] for point in curve],
            color="#4F46E5",
        )
        axis.axvline(champion["optimal_threshold"], color="#EF4444", linestyle="--", label="Optimal threshold")
        axis.legend()
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Expected cost (1·FP + 5·FN)")
    axis.set_title("Cost-sensitive threshold curve")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
