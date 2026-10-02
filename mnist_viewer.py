"""Load MNIST and illustrate binary classification of handwritten digits."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier


class MnistViewer:
    """Prepare MNIST and visualize a sample with its binary prediction."""

    def __init__(self) -> None:
        """Initialize the dataset and classifier."""
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.X_train: np.ndarray | None = None
        self.y_train_5: np.ndarray | None = None
        self.model: SGDClassifier | None = None

    def load_data(self) -> None:
        """Fetch MNIST version 1 and preserve the original sample order."""
        mnist = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            parser="auto",
        )
        self.X = mnist.data
        self.y = mnist.target.astype(str)
        print(f"Loaded {len(self.y):,} images with {self.X.shape[1]} pixels each.")

    def prepare_data(self) -> pd.DataFrame:
        """Use the first 60,000 images for training and create binary labels."""
        if self.X is None or self.y is None:
            raise RuntimeError("Call load_data() first.")

        self.X_train = self.X[:60000]
        self.y_train_5 = self.y[:60000] == "5"

        return pd.DataFrame(
            {
                "sample_index": np.arange(5),
                "original_label": self.y[:5],
                "is_five": self.y_train_5[:5],
            }
        )

    def get_sample(self, index: int) -> tuple[np.ndarray, str]:
        """Return the pixel values and original label at a zero-based index."""
        if self.X is None or self.y is None:
            raise RuntimeError("Call load_data() first.")
        if not 0 <= index < len(self.y):
            raise IndexError("The sample index is outside the dataset.")

        return self.X[index], str(self.y[index])

    def train_model(self) -> None:
        """Train the same binary SGD classifier used in the original notebook."""
        if self.X_train is None or self.y_train_5 is None:
            raise RuntimeError("Call prepare_data() first.")

        self.model = SGDClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train_5)
        print("Training complete: the model predicts whether a digit is 5.")

    def plot_digit(self, index: int = 2) -> go.Figure:
        """Display a digit, its actual label, and the model's prediction."""
        if self.model is None:
            raise RuntimeError("Call train_model() first.")

        image, label = self.get_sample(index)
        prediction = bool(self.model.predict(image.reshape(1, -1))[0])
        meaning = "5" if prediction else "not 5"

        print(f"Sample index: {index}")
        print(f"Actual digit: {label}")
        print(f"Actual binary target (is 5): {label == '5'}")
        print(f"Predicted binary target: {prediction} ({meaning})")

        figure = px.imshow(
            image.reshape(28, 28),
            color_continuous_scale="gray_r",
            zmin=0,
            zmax=255,
            origin="upper",
            title=(
                f"MNIST sample X[{index}] | Actual digit: {label}"
                f"<br>Prediction: {prediction} ({meaning})"
            ),
            labels={"color": "Pixel intensity"},
        )
        figure.update_layout(
            width=550,
            height=550,
            coloraxis_showscale=False,
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        return figure