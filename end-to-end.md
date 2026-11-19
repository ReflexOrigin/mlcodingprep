# General Machine Learning Classification Pipeline

## 1. Leakage-Free Ingestion & Splitting

The first step in any machine learning pipeline is to **load, inspect, and split the dataset before applying preprocessing transformations**.

This prevents **data leakage** and ensures that model performance is evaluated on genuinely unseen data.

### 1.1 Import Dependencies

```python
import pandas as pd
from sklearn.model_selection import train_test_split
```

### 1.2 Load Dataset

```python
df = pd.read_csv("data.csv")
```

### 1.3 Initial Sanity Checks

Before splitting, inspect the dataset's structure, missing values, and target distribution.

```python
# Dataset dimensions
print(f"Shape: {df.shape}")

# Missing values
print(df.isnull().sum()[df.isnull().sum() > 0])

# Target class distribution (classification)
print(df["target"].value_counts(normalize=True))
```

**What to examine:**

- **Shape:** Number of observations and features.
- **Missing Values:** Identify columns requiring imputation.
- **Class Distribution:** Detect potential class imbalance.
- **Target Integrity:** Verify that the target column is correctly defined.

### 1.4 Feature-Target Separation

Separate the independent variables (`X`) from the target variable (`y`).

```python
X = df.drop(columns=["target"])
y = df["target"]
```

### 1.5 Leakage-Free Train-Test Split

Split the dataset **before fitting any data-dependent preprocessing transformations**, such as imputation, scaling, encoding, or feature selection.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Preserve class distribution for classification
)
```

> **Note:** For time-dependent data, use chronological splitting. For repeated observations belonging to the same entity, consider group-aware splitting to prevent cross-group leakage. Stratification is appropriate for many ordinary classification datasets but is not universally appropriate.

---

## 2. Modular Preprocessing Pipeline

A robust machine learning pipeline should handle numerical and categorical features separately using **modular, reusable preprocessing components**.

Scikit-learn's `Pipeline` and `ColumnTransformer` allow multiple preprocessing operations to be combined into a single workflow, ensuring consistency between training and inference.

### 2.1 Import Dependencies

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
```

### 2.2 Identify Feature Types

Dynamically identify numerical and categorical columns from the training dataset.

```python
# Numerical features
num_cols = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

# Categorical features
cat_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print(f"Numerical Features: {len(num_cols)}")
print(f"Categorical Features: {len(cat_cols)}")
```

**Why dynamic detection?**

- Eliminates manual feature classification.
- Makes the pipeline reusable across datasets.
- Reduces hardcoded dependencies.
- Supports changes in dataset structure.

> **Note:** Numeric columns are not necessarily continuous variables. Integer-encoded categories and identifiers may require separate treatment. Boolean, datetime, and other unsupported data types should also be handled explicitly.

### 2.3 Numerical Preprocessing Pipeline

Numerical features require appropriate handling of missing values and feature scaling.

```python
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

### 2.4 Categorical Preprocessing Pipeline

Categorical features require imputation and encoding. Unknown categories may appear at inference time, so the encoder should handle them safely.

```python
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
```

### 2.5 Combine Feature Transformations

Use `ColumnTransformer` to apply each preprocessing pipeline to its corresponding feature subset.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ],
    remainder="drop"
)
```

> **Leakage prevention:** Do not fit `preprocessor` on the entire dataset. It will be fitted on `X_train` automatically when the model pipeline is trained.

---

## 3. Baseline First → Strong Candidate Model

A reliable machine learning workflow should begin with a **simple, interpretable baseline model** before introducing more complex algorithms.

This establishes a performance reference and helps determine whether additional model complexity provides meaningful improvements.

The objective is not simply to achieve the highest score, but to evaluate the trade-offs between **predictive performance, generalization, interpretability, and computational complexity**.

### 3.1 Import Dependencies

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
```

### 3.2 Baseline Model — Logistic Regression

Logistic Regression provides a strong baseline for classification problems due to its simplicity, efficiency, and interpretability.

```python
baseline_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# Train baseline model
baseline_pipe.fit(X_train, y_train)
```

> **Note:** The `class_weight="balanced"` setting is useful to test when classes are imbalanced; it is not automatically optimal for every dataset. Choose models and thresholds using cross-validation or a separate validation set, and reserve the test set for final evaluation.

---

## 4. Evaluation & Error Analysis

Model evaluation should go beyond a single performance metric to understand **predictive quality, generalization, class imbalance, and error patterns**.

A reliable evaluation framework should measure both overall predictive performance and the consequences of different types of classification errors.

The objective is not simply to identify the highest-scoring model, but to understand **where, why, and under what conditions the model fails**.

### 4.1 Import Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
```

### 4.2 Reusable Model Evaluation Function

Create a reusable evaluation function to assess different classification models using consistent metrics.

The following implementation assumes **binary classification with labels 0 and 1**, where `1` represents the positive class.

```python
def evaluate_model(model, X_val, y_val, model_name="Model"):

    # Generate predictions
    y_pred = model.predict(X_val)

    # Initialize probability-based metrics
    roc = None
    pr_auc = None
    y_prob = None

    # Extract positive-class probabilities
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)

        if 1 in classes:
            positive_idx = classes.index(1)
            y_prob = model.predict_proba(X_val)[:, positive_idx]

    print(f"\n{'=' * 20} {model_name} {'=' * 20}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_val,
        y_pred,
        digits=4,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    print(f"\nConfusion Matrix:\n{cm}")

    # Probability-based metrics
    if y_prob is not None and len(np.unique(y_val)) == 2:
        roc = roc_auc_score(y_val, y_prob)
        pr_auc = average_precision_score(y_val, y_prob)

        print(f"\nROC-AUC: {roc:.4f}")
        print(f"Average Precision (AP): {pr_auc:.4f}")

    # Return evaluation metrics
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(
            y_val, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_val, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_val, y_pred, zero_division=0
        ),
        "roc_auc": roc,
        "pr_auc": pr_auc
    }
```

### 4.3 Example: Evaluate the Baseline

After selecting a model using training data and an appropriate validation strategy, use the held-out test set for a final evaluation.

```python
baseline_metrics = evaluate_model(
    baseline_pipe,
    X_test,
    y_test,
    model_name="Logistic Regression Baseline"
)

print(pd.DataFrame([baseline_metrics]))
```
