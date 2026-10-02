# Performance Metrics for Classification

This notebook provides an interactive introduction to **performance metrics used in evaluating classification models**. It is adapted from Aurélien Géron's Chapter 3 notebook under the Apache v2 License.

## Purpose

The focus is on **evaluating classifiers**, not on training them. Students will explore key metrics used to assess classification performance, such as:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve and AUC

## Learning Format

This notebook is designed for **active learning** in a classroom setting. Students are encouraged to:

- Run code cells during instruction
- Respond to questions marked "_To the student_" in new cells
- Take notes and annotate the notebook
- Commit and push their personalized version to their own repository

## What is Classification?

Classification is a supervised learning task where the goal is to assign input data to one of several predefined categories. A **classifier** is an algorithm or function that performs this mapping from input features to class labels.

Examples include:
- Email spam detection
- Medical diagnosis
- Image recognition

For more background, see the Wikipedia article on classification and classification algorithms.

## Environment setup

Create and activate a virtual environment from the project root:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, use `python3 -m venv .venv` and activate it with
`source .venv/bin/activate`.

Install the project dependencies and register the environment as a Jupyter
kernel:

```shell
python -m pip install -r requirements.txt
python -m ipykernel install --user --name performance-metrics-classification --display-name "Python (Performance Metrics Classification)"
```

Use this environment to run Python scripts with `python path/to/script.py`, or
select **Python (Performance Metrics Classification)** as the kernel in
Jupyter.
