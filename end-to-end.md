## 1. Leakage-Free Ingestion & Splitting

```python
import pandas as pd
from sklearn.model_selection import train_test_split
# 1. Load data
df = pd.read_csv("data.csv")
# 2. Quick sanity checks
print(f"Shape: {df.shape}")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(df['target'].value_counts(normalize=True))  # Check class imbalance
# 3. Leak-free split FIRST
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify for classification
)
```

## 2. Modular Preprocessing Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Separate feature types dynamically
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Numerical: median impute + standard scaling
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: frequent impute + one-hot (handle_unknown='ignore' prevents inference crashes!)
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Full preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])
```

## 3. Baseline First → Strong Candidate Model

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. Baseline Model Pipeline
baseline_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
baseline_pipe.fit(X_train, y_train)

# 2. Candidate Model Pipeline (e.g., Random Forest or Gradient Boosting)
candidate_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1))
])
candidate_pipe.fit(X_train, y_train)
```

## 4. Evaluation & Error Analysis

```python
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix

def evaluate_model(model, X_val, y_val, model_name="Model"):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
    
    print(f"\n=================== {model_name} Evaluation ===================")
    print(classification_report(y_val, y_pred, digits=4))
    
    if y_prob is not None:
        roc = roc_auc_score(y_val, y_prob)
        pr_auc = average_precision_score(y_val, y_prob)
        print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f}")
    
    cm = confusion_matrix(y_val, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    return {"roc_auc": roc, "pr_auc": pr_auc}

eval_base = evaluate_model(baseline_pipe, X_test, y_test, "Baseline Logistic Regression")
eval_cand = evaluate_model(candidate_pipe, X_test, y_test, "Candidate Random Forest")
```

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
