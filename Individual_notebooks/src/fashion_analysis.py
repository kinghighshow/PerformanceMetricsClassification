import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict, cross_validate


class FashionAnalysis:
    """Helper methods for the Fashion-MNIST section of the assignment."""

    @staticmethod
    def plot_samples(X, y, rows=3, cols=5, random_state=42):
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X), size=rows * cols, replace=False)

        fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
        axes = np.asarray(axes).ravel()
        for ax, idx in zip(axes, indices):
            ax.imshow(np.asarray(X[idx]).reshape(28, 28), cmap="gray")
            ax.set_title(f"Label: {y[idx]}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def make_binary_targets(y_train, y_test, positive_class="7"):
        y_train_binary = np.asarray(y_train) == positive_class
        y_test_binary = np.asarray(y_test) == positive_class
        return y_train_binary, y_test_binary

    @staticmethod
    def train_sgd(X_train, y_train_binary, random_state=42):
        clf = SGDClassifier(random_state=random_state)
        clf.fit(X_train, y_train_binary)
        return clf

    @staticmethod
    def evaluate_model(clf, X_train, y_train_binary, cv=3):
        cv_results = cross_validate(
            estimator=clf,
            X=X_train,
            y=y_train_binary,
            cv=cv,
            scoring="accuracy",
        )
        y_pred = cross_val_predict(
            clf, X_train, y_train_binary, cv=cv
        )
        cm = confusion_matrix(y_train_binary, y_pred)
        return cv_results, y_pred, cm

    @staticmethod
    def evaluate_dummy(X_train, y_train_binary, cv=3):
        dummy = DummyClassifier(strategy="most_frequent")
        results = cross_validate(
            dummy,
            X_train,
            y_train_binary,
            cv=cv,
            scoring="accuracy",
        )
        return dummy, results

    @staticmethod
    def plot_confusion_matrix(cm, title="Fashion-MNIST Confusion Matrix"):
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        plt.xlabel("Predicted label")
        plt.ylabel("Actual label")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_mnist_confusion_matrix(cm):
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Not 5", "5"],
            yticklabels=["Not 5", "5"],
        )
        plt.xlabel("Predicted label")
        plt.ylabel("Actual label")
        plt.title("MNIST Confusion Matrix")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def summarize_binary_metrics(cm):
        tn, fp, fn, tp = cm.ravel()
        precision = precision_score([0] * (tn + fp) + [1] * (fn + tp),
                                    [0] * tn + [1] * fp + [0] * fn + [1] * tp)
        recall = recall_score([0] * (tn + fp) + [1] * (fn + tp),
                              [0] * tn + [1] * fp + [0] * fn + [1] * tp)
        f1 = f1_score([0] * (tn + fp) + [1] * (fn + tp),
                      [0] * tn + [1] * fp + [0] * fn + [1] * tp)
        return {
            "accuracy": (tn + tp) / cm.sum(),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
