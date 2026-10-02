# Performance Metrics Classification Workshop

**Final_Project_Group 1** | Use case: MNIST image classification

This repository contains our work for the Performance Metrics Classification workshop. We worked through the instructor's notebook (adapted from Géron's chapter 3), `PerformanceMetricsClassification.ipynb`, and answered every "To the student" talking point. Each member wrote their part in an individual notebook, and `Collated_Team_Notebook.ipynb` brings the four parts together in one notebook.

## Team

| Member | Talking points | Helper class |
|---|---|---|
| Antonio | Classification and classifier definitions, the Boolean target and unchanged `X`, `some_digit` and its prediction, precision vs recall real-world cases | `MnistViewer` |
| Sultan | `cross_validate`, accuracy, `DummyClassifier`, cross-validation fold research | `CrossValReport` |
| Eche | Fashion-MNIST exercise, confusion matrix | `FashionAnalysis` |
| John | Security drone exercise, recall in code, F1 in code, threshold behaviour | `MetricsCalculator` |

## Repository structure

```
PerformanceMetricsClassification/
├── PerformanceMetricsClassification.ipynb   # instructor's notebook
├── Collated_Team_Notebook.ipynb             # our collated answers (final notebook)
├── README.md
├── requirements.txt
├── .gitignore
└── Individual_notebooks/
    ├── Antonio_Notebook.ipynb               # each person's original working notebook
    ├── Sultan_Notebook.ipynb
    ├── Eche_Notebook.ipynb
    ├── John_Notebook.ipynb
    └── src/
        ├── mnist_viewer.py                  # MnistViewer
        ├── cross_val_report.py              # CrossValReport
        ├── fashion_analysis.py              # FashionAnalysis
        └── metrics_calculator.py            # MetricsCalculator
```

The four individual notebooks are kept as the record of who wrote what. The collated notebook combines them under one header per person (`## Antonio`, `## Sultan`, `## Eche`, `## John`) so the answers read in order.

## How to run it

1. Create and activate a virtual environment, then install the dependencies:
   ```
   python -m venv .venv
   pip install -r requirements.txt
   ```
2. Start Jupyter from the repository root and open `Collated_Team_Notebook.ipynb`.
3. Run all cells from the top. The setup cell imports the helper classes from `Individual_notebooks/src/` (as `Individual_notebooks.src.<module>`), so no extra setup is needed. The notebook does not depend on any variable from the instructor's notebook.

The notebook downloads MNIST and Fashion-MNIST from OpenML the first time it runs, so it needs an internet connection and takes a little while.

## How the code is organised

All reusable code is written as class methods in `Individual_notebooks/src/`, one file per person. The notebooks only import the classes and call their methods, and the notebook cells hold the questions, the answers and the calls that produce the numbers and charts. This also meant we could work in parallel without editing the same file.

## What we did

### 1. Classification, preprocessing and training (Antonio)

- A classification problem asks which predefined category an observation belongs to based on its features. A classifier is the algorithm or function that maps the features to a class label. Here the features are the 784 pixel values of an image, and we turn the ten-class problem into a binary one: "is this digit a 5?"
- `y_train_5` holds `True` where the original label is "5" and `False` for every other digit. `X` is not changed because the pixels are still the input for the new question; only the question, and therefore the target, changed.
- `some_digit` is `X[2]`, the third image, because it is the last assignment in the notebook before the prediction. Its true label is 4, and the model predicts `False` (not a 5), which is a correct rejection.
- Precision vs recall depends on what a mistake costs. High recall suits a first-stage medical screen or a safety defect check, where a miss is the expensive error. High precision suits shortlisting investments or auto-deleting spam, where a false alarm is the expensive error.

### 2. Cross-validation, accuracy and the baseline (Sultan)

- `cv=3` splits the 60,000 training images into 3 folds of 20,000. The model is trained 3 times, each time on 40,000 images and tested on the remaining 20,000. `fit_time`, `score_time` and `test_score` each have one value per fold.
- In the first fold the model classified 19,007 of 20,000 digits correctly and 993 incorrectly (accuracy 95.0%).
- A handwritten rule that always answers "not 5" already gets about 90% accuracy, because roughly nine in ten digits are not 5. The `DummyClassifier` behaves the same way and scores 90.97% on every fold. The `SGDClassifier` averages about 95.7%.
- The point of the comparison is that accuracy alone flatters a model on an imbalanced target. The `SGDClassifier` has a precision of 0.84 but a recall of only 0.65, so it still misses a third of the 5s.
- A fold is one of the k parts the data is divided into. Each part is used once for testing while the others train the model, and the scores are averaged.

### 3. Fashion-MNIST and the confusion matrix (Eche)

- We repeated the whole process on Fashion-MNIST with Sneaker (label 7) as the positive class: fetch the data, split 60,000 / 10,000, build the binary target, train an `SGDClassifier`, then evaluate with 3-fold cross-validation, a `DummyClassifier` baseline and a confusion matrix.
- The process carries over unchanged because the data has the same shape as MNIST (784 pixels per image, one label each). The `SGDClassifier` scores about 97.3% against 90% for the baseline.
- Fashion-MNIST confusion matrix: 52,767 true negatives, 1,233 false positives, 409 false negatives and 5,591 true positives. From those counts, precision is about 0.82 and recall about 0.93.
- `confusion_matrix` needs `y_true` and `y_pred`. For a binary problem it returns a 2 × 2 array laid out as `[[TN, FP], [FN, TP]]`, with actual classes in the rows and predicted classes in the columns. We explain every value in both matrices from the original notebook, including the perfect-predictions matrix where both error cells are zero.

### 4. Precision, recall, F1 and thresholds (John)

- Security drone: of 12 alarms, 8 were correct, so precision is 8 / 12 = 66.7%. Of 10 real intrusions, 8 were detected, so recall is 8 / 10 = 80%. The drone catches most intruders but raises false alarms (shadows, falling boxes).
- Recall in code from the MNIST confusion matrix: 3,530 / (3,530 + 1,891) = 0.6512.
- F1 from TP, FP and FN: (2 × 3,530) / (2 × 3,530 + 687 + 1,891) = 0.7325, which matches `f1_score` from scikit-learn.
- Raising the decision threshold makes the classifier stricter, which usually raises precision and lowers recall. Lowering it does the opposite. The threshold is a dial between "only flag what I am sure about" and "catch as many positives as possible". The notebook plots precision and recall against the threshold with the chosen threshold marked.

## Key results

| Experiment | Metric | Value |
|---|---|---|
| MNIST, 5 vs rest | `SGDClassifier` accuracy (3-fold average) | about 95.7% |
| MNIST, 5 vs rest | `DummyClassifier` accuracy | 90.97% |
| MNIST, 5 vs rest | Precision / Recall / F1 of `SGDClassifier` | 0.84 / 0.65 / 0.7325 |
| Fashion-MNIST, Sneaker vs rest | `SGDClassifier` accuracy (3-fold average) | about 97.3% |
| Fashion-MNIST, Sneaker vs rest | `DummyClassifier` accuracy | 90.0% |
| Security drone | Precision / Recall | 66.7% / 80% |

## Summary of insights

1. A baseline is essential. A model that never detects the positive class can still score about 90% accuracy, so a good accuracy number means little until it is compared with a `DummyClassifier`.
2. The confusion matrix is the foundation. Precision, recall and F1 all come from TP, FP and FN, and each one answers a different question about the errors.
3. Precision and recall pull against each other. Moving the threshold improves one at the cost of the other, and F1 summarises the balance in a single number.
4. The right metric depends on the cost of the mistake, not on the algorithm. Recall is the priority when a miss is dangerous; precision is the priority when a false alarm is costly.
5. The same workflow generalises. Applying it to Fashion-MNIST needed no new method, only a different positive class.

## Workshop requirements checklist

| Requirement | Where it is covered |
|---|---|
| Remote repository named "PerformanceMetricsClassification" | This repository |
| Instructor's repository cloned and notebook worked through | `PerformanceMetricsClassification.ipynb` (instructor's notebook, kept in the repo) |
| Every "To the student" talking point answered (13 hits plus the folds research prompt) | `Collated_Team_Notebook.ipynb`; the opening table lists each talking point and who answered it |
| Section answering all questions and challenges | `Collated_Team_Notebook.ipynb`, with one header per team member |
| Notebooks copied to our own repository | Root of this repository and `Individual_notebooks/` |
| README with an introduction and a summary of insights | This file |
| Coding standards and best practices | Helper logic is in class methods under `Individual_notebooks/src/`; notebooks only import and call them |
| One PDF with the workshop title, names and a link to this repository | Submitted separately before the deadline |
