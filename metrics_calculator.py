"""Utilities for calculating binary classification performance metrics."""

import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_curve


class MetricsCalculator:
    """Calculate and visualize binary classification metrics."""

    @staticmethod
    def get_confusion_counts(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> tuple[int, int, int, int]:
        """Return TN, FP, FN and TP from a binary confusion matrix."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return int(tn), int(fp), int(fn), int(tp)

    @staticmethod
    def precision(tp: int, fp: int) -> float:
        """Calculate precision from true and false positive counts."""
        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    @staticmethod
    def recall(tp: int, fn: int) -> float:
        """Calculate recall from true positive and false negative counts."""
        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    @staticmethod
    def f1(tp: int, fp: int, fn: int) -> float:
        """Calculate the F1 score directly from TP, FP and FN."""
        denominator = (2 * tp) + fp + fn

        if denominator == 0:
            return 0.0

        return (2 * tp) / denominator

    @staticmethod
    def plot_precision_recall_threshold(
        y_true: np.ndarray,
        decision_scores: np.ndarray,
        chosen_threshold: float = 0.0
    ) -> go.Figure:
        """Plot precision and recall against the decision threshold."""
        precision, recall, thresholds = precision_recall_curve(
            y_true,
            decision_scores
        )

        # Find the calculated threshold closest to the chosen threshold.
        marker_index = int(
            np.argmin(np.abs(thresholds - chosen_threshold))
        )

        marker_threshold = thresholds[marker_index]
        marker_precision = precision[marker_index]
        marker_recall = recall[marker_index]

        figure = go.Figure()

        figure.add_trace(
            go.Scatter(
                x=thresholds,
                y=precision[:-1],
                mode="lines",
                name="Precision"
            )
        )

        figure.add_trace(
            go.Scatter(
                x=thresholds,
                y=recall[:-1],
                mode="lines",
                name="Recall"
            )
        )

        # Mark precision at the chosen threshold.
        figure.add_trace(
            go.Scatter(
                x=[marker_threshold],
                y=[marker_precision],
                mode="markers",
                marker={"size": 12},
                name="Precision at chosen threshold"
            )
        )

        # Mark recall at the chosen threshold.
        figure.add_trace(
            go.Scatter(
                x=[marker_threshold],
                y=[marker_recall],
                mode="markers",
                marker={"size": 12},
                name="Recall at chosen threshold"
            )
        )

        figure.add_vline(
            x=chosen_threshold,
            line_dash="dash",
            annotation_text=f"Threshold = {chosen_threshold:g}"
        )

        figure.update_layout(
            title="Precision and Recall vs Decision Threshold",
            xaxis_title="Decision Threshold",
            yaxis_title="Score",
            yaxis_range=[0, 1]
        )

        return figure

    @staticmethod
    def prepare_mnist_threshold_data() -> tuple[np.ndarray, np.ndarray]:
        """Create MNIST binary labels and decision scores for threshold analysis."""
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)

        x_train = mnist.data[:60000]
        y_train = mnist.target[:60000]

        y_train_5 = y_train == "5"

        classifier = SGDClassifier(random_state=42)

        y_scores = cross_val_predict(
            classifier,
            x_train,
            y_train_5,
            cv=3,
            method="decision_function"
        )

        return y_train_5, y_scores