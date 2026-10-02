import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate


class CrossValReport:
    """Runs 3-fold CV for SGDClassifier and DummyClassifier on MNIST ('5' vs rest)."""

    def __init__(self, cv=3, random_state=42):
        self.cv = cv
        self.random_state = random_state
        self.sgd_clf = SGDClassifier(random_state=random_state)
        self.dummy_clf = DummyClassifier()
        self.X_train = None
        self.y_train_5 = None
        self.sgd_results = None
        self.dummy_results = None



    def load_data(self):
        mnist = fetch_openml("mnist_784", as_frame=False, parser="auto")
        X, y = mnist.data, mnist.target
        self.X_train, y_train = X[:60000], y[:60000]
        self.y_train_5 = (y_train == "5")
        return self

    def run(self):
        if self.X_train is None:
            self.load_data()
        self.sgd_results = cross_validate(
            self.sgd_clf, self.X_train, self.y_train_5, cv=self.cv, scoring="accuracy")
        self.dummy_results = cross_validate(
            self.dummy_clf, self.X_train, self.y_train_5, cv=self.cv, scoring="accuracy")
        return self

    def results_table(self):
        """Per-fold accuracy for both classifiers."""
        n = len(self.sgd_results["test_score"])
        return pd.DataFrame({
            "fold": [f"Fold {i + 1}" for i in range(n)],
            "SGDClassifier": self.sgd_results["test_score"],
            "DummyClassifier": self.dummy_results["test_score"],
        })

    def describe_fold(self, fold=0, results="sgd"):
        """Plain-English description of one fold's accuracy."""
        res = self.sgd_results if results == "sgd" else self.dummy_results
        n_test = len(self.y_train_5) // self.cv
        acc = res["test_score"][fold]
        correct = round(acc * n_test)
        return (f"Fold {fold + 1}: out of {n_test:,} digits, {correct:,} were correctly "
                f"predicted, while {n_test - correct:,} were incorrectly predicted "
                f"(accuracy = {acc:.2%}).")

    def plot_accuracy_per_fold(self):
        """Grouped bar chart: SGD next to Dummy, per fold (Plotly)."""
        df = self.results_table()
        fig = go.Figure()
        fig.add_bar(x=df["fold"], y=df["SGDClassifier"], name="SGDClassifier",
                    marker_color="#4C78FF", text=df["SGDClassifier"].round(4),
                    textposition="outside")
        fig.add_bar(x=df["fold"], y=df["DummyClassifier"], name="DummyClassifier",
                    marker_color="#FF7F50", text=df["DummyClassifier"].round(4),
                    textposition="outside")
        fig.update_layout(
            barmode="group", template="plotly_white",
            title="Accuracy per cross-validation fold (5 vs. not-5)",
            xaxis_title="Fold", yaxis_title="Accuracy",
            yaxis=dict(range=[0.85, 1.0], tickformat=".0%"),
            legend_title_text="Classifier")
        return fig
