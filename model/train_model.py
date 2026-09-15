
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "loan_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
FI_PATH = os.path.join(BASE_DIR, "feature_importance.json")

df = pd.read_csv(DATA_PATH)

TARGET = "Loan_Status"
y = (df[TARGET] == "Y").astype(int)
X = df.drop(columns=[TARGET])

numeric_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Credit_Score",
    "Age",
    "Existing_Loans",
    "Debt_to_Income",
]
categorical_features = [
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
}

print("Model performance on held-out test set:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

# --- Feature importance (map one-hot columns back to friendly names) ---
ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_names = list(ohe.get_feature_names_out(categorical_features))
all_feature_names = numeric_features + cat_names
importances = pipeline.named_steps["classifier"].feature_importances_

fi_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False).head(10)
fi_df["importance"] = fi_df["importance"].round(4)

with open(FI_PATH, "w") as f:
    json.dump(fi_df.to_dict(orient="records"), f, indent=2)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print(f"\nSaved trained pipeline to {MODEL_PATH}")
print(f"Saved metrics to {METRICS_PATH}")
print(f"Saved feature importance to {FI_PATH}")
