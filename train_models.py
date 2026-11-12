import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Columns where 0 represents a missing clinical measurement rather than a true zero
ZERO_AS_MISSING_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def load_or_fetch_dataset() -> pd.DataFrame:
    csv_path = DATA_DIR / "diabetes.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    print("Fetching Pima Indians Diabetes dataset from OpenML...")
    dataset = fetch_openml(data_id=37, as_frame=True, parser="auto")
    df = dataset.frame.copy()

    rename_map = {
        "preg": "Pregnancies",
        "plas": "Glucose",
        "pres": "BloodPressure",
        "skin": "SkinThickness",
        "insu": "Insulin",
        "mass": "BMI",
        "pedi": "DiabetesPedigreeFunction",
        "age": "Age",
    }
    df = df.rename(columns=rename_map)
    df["Outcome"] = (df["class"] == "tested_positive").astype(int)
    df = df[FEATURE_COLUMNS + ["Outcome"]]

    df.to_csv(csv_path, index=False)
    print(f"Saved dataset to {csv_path} ({len(df)} records).")
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "clinical_impute",
                SimpleImputer(missing_values=0, strategy="median"),
                ZERO_AS_MISSING_COLS,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def train_and_evaluate() -> None:
    df = load_or_fetch_dataset()
    X = df[FEATURE_COLUMNS]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classifiers = {
        "decision_tree": {
            "label": "Decision Tree",
            "estimator": DecisionTreeClassifier(max_depth=5, random_state=42),
        },
        "ann": {
            "label": "Neural Network (MLP)",
            "estimator": MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=600, random_state=42
            ),
        },
        "naive_bayes": {
            "label": "Naive Bayes",
            "estimator": GaussianNB(),
        },
        "knn": {
            "label": "K-Nearest Neighbors",
            "estimator": KNeighborsClassifier(n_neighbors=7),
        },
        "logistic_regression": {
            "label": "Logistic Regression",
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
        },
        "svm": {
            "label": "Support Vector Machine",
            "estimator": SVC(probability=True, kernel="rbf", random_state=42),
        },
    }

    metrics_summary = {}

    for key, config in classifiers.items():
        pipeline = Pipeline(
            steps=[
                ("imputer", build_preprocessor()),
                ("scaler", StandardScaler()),
                ("classifier", config["estimator"]),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics_summary[key] = {
            "name": config["label"],
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        }

        artifact_path = MODELS_DIR / f"{key}.joblib"
        joblib.dump(pipeline, artifact_path)
        print(
            f"[{config['label']}] Accuracy: {metrics_summary[key]['accuracy']:.2%} | "
            f"ROC-AUC: {metrics_summary[key]['roc_auc']:.4f}"
        )

    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"\nAll 6 model pipelines and metrics saved to '{MODELS_DIR}/'.")


if __name__ == "__main__":
    train_and_evaluate()