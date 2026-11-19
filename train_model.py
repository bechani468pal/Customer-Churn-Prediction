import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "dataset" / "customer_churn.csv")

print("Original Dataset Shape:", df.shape)

# Remove customer ID
df.drop(columns=["customerID"], inplace=True)

# Convert TotalCharges into numeric values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Remove duplicate records
df.drop_duplicates(inplace=True)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:", df.shape)

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

# Feature 1: Tenure Group
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, float("inf")],
    labels=["New", "Medium", "Long", "Loyal"]
)

# Feature 2: Average Monthly Spend
df["AverageMonthlySpend"] = (
    df["TotalCharges"] /
    df["tenure"].replace(0, np.nan)
)

# Feature 3: Total Services
service_columns = [
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["TotalServices"] = (
    df[service_columns] == "Yes"
).sum(axis=1)

print("\nNew Features Created:")
print(df[[
    "TenureGroup",
    "AverageMonthlySpend",
    "TotalServices"
]].head())

# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Identify numerical columns
numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Identify categorical columns
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ==========================================
# 4. DATA PREPROCESSING
# ==========================================

# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both pipelines
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

print("\nPreprocessing Pipeline Created!")

# ==========================================
# 5. DEFINE THREE ML MODELS
# ==========================================

models = {
    "Logistic Regression": (
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        {
            "model__C": [0.1, 1, 10]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        {
            "model__n_estimators": [100],
            "model__max_depth": [5, 10]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        {
            "model__max_depth": [5, 10, None]
        }
    )
}

results = {}

# ==========================================
# 6. TRAIN AND TUNE MODELS
# ==========================================

for name, (model, params) in models.items():

    print("\nTraining Model:", name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        pipeline,
        params,
        cv=3,
        scoring="roc_auc",
        n_jobs=1
    )

    grid.fit(X_train, y_train)

    results[name] = {
        "model": grid.best_estimator_,
        "cv_auc": grid.best_score_,
        "params": grid.best_params_
    }

    print("Best Parameters:", grid.best_params_)
    print("Cross Validation AUC:",
          round(grid.best_score_, 4))

# ==========================================
# 7. MODEL EVALUATION
# ==========================================

print("\n========== MODEL EVALUATION ==========")

for name, result in results.items():

    model = result["model"]

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probabilities)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    pr_auc = average_precision_score(
        y_test, probabilities
    )

    result["test_auc"] = auc

    print("\nModel:", name)
    print("ROC-AUC:", round(auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("PR-AUC:", round(pr_auc, 4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

# ==========================================
# 8. SELECT BEST MODEL
# ==========================================

best_name = max(
    results,
    key=lambda name: results[name]["cv_auc"]
)

best_model = results[best_name]["model"]

print("\n================================")
print("BEST MODEL:", best_name)
print("Cross Validation AUC:",
      round(results[best_name]["cv_auc"], 4))
print("Test AUC:",
      round(results[best_name]["test_auc"], 4))
print("================================")

# ==========================================
# 9. SAVE BEST MODEL
# ==========================================

model_path = BASE_DIR / "models" / "model.pkl"

joblib.dump(
    {
        "model": best_model,
        "threshold": 0.5,
        "model_name": best_name
    },
    model_path
)

print("\nModel Saved Successfully!")
print("Saved Location:", model_path)

print("\nCUSTOMER CHURN PIPELINE COMPLETED!")