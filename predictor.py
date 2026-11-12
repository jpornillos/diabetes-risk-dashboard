import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pydantic import BaseModel, Field

MODELS_DIR = Path("models")

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


class PatientInput(BaseModel):
    pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    glucose: float = Field(..., ge=0, le=400, description="Plasma glucose (mg/dL)")
    blood_pressure: float = Field(..., ge=0, le=250, description="Diastolic BP (mm Hg)")
    skin_thickness: float = Field(..., ge=0, le=120, description="Triceps skin fold (mm)")
    insulin: float = Field(..., ge=0, le=1000, description="2-Hour serum insulin (mu U/ml)")
    bmi: float = Field(..., ge=0, le=80, description="Body mass index (kg/m²)")
    diabetes_pedigree: float = Field(..., ge=0.0, le=3.0, description="Diabetes pedigree function")
    age: int = Field(..., ge=1, le=120, description="Age in years")


class DiabetesPredictor:
    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.metrics = self._load_metrics()
        self.pipelines = self._load_pipelines()

    def _load_metrics(self) -> dict[str, Any]:
        metrics_file = self.models_dir / "metrics.json"
        if not metrics_file.exists():
            raise FileNotFoundError(
                f"Metrics file not found at {metrics_file}. Run train_models.py first."
            )
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_pipelines(self) -> dict[str, Any]:
        loaded = {}
        for key in self.metrics:
            model_path = self.models_dir / f"{key}.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Model artifact missing: {model_path}")
            loaded[key] = joblib.load(model_path)
        return loaded

    def predict_all(self, patient: PatientInput) -> dict[str, Any]:
        # Map Pydantic fields to the exact DataFrame column names used during training
        input_df = pd.DataFrame(
            [
                {
                    "Pregnancies": patient.pregnancies,
                    "Glucose": patient.glucose,
                    "BloodPressure": patient.blood_pressure,
                    "SkinThickness": patient.skin_thickness,
                    "Insulin": patient.insulin,
                    "BMI": patient.bmi,
                    "DiabetesPedigreeFunction": patient.diabetes_pedigree,
                    "Age": patient.age,
                }
            ],
            columns=FEATURE_COLUMNS,
        )

        results = []
        positive_votes = 0
        probabilities = []

        for key, pipeline in self.pipelines.items():
            meta = self.metrics[key]
            pred_class = int(pipeline.predict(input_df)[0])
            prob_positive = float(pipeline.predict_proba(input_df)[0][1])
            prob_percent = round(prob_positive * 100, 1)

            if pred_class == 1:
                positive_votes += 1
            probabilities.append(prob_percent)

            results.append(
                {
                    "key": key,
                    "name": meta["name"],
                    "prediction": pred_class,
                    "label": "Diabetic Risk" if pred_class == 1 else "Low Risk",
                    "probability": prob_percent,
                    "accuracy": round(meta["accuracy"] * 100, 1),
                    "roc_auc": meta["roc_auc"],
                }
            )

        # Sort models by ROC-AUC descending so strongest models appear first
        results.sort(key=lambda item: item["roc_auc"], reverse=True)

        avg_prob = round(sum(probabilities) / len(probabilities), 1)
        total_models = len(results)

        if avg_prob >= 50.0 or positive_votes >= 4:
            risk_tier = "High Risk"
        elif avg_prob >= 35.0 or positive_votes >= 2:
            risk_tier = "Moderate Risk"
        else:
            risk_tier = "Low Risk"

        return {
            "consensus": {
                "risk_tier": risk_tier,
                "average_probability": avg_prob,
                "positive_votes": positive_votes,
                "total_models": total_models,
            },
            "models": results,
        }