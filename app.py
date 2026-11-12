from flask import Flask, jsonify, render_template, request
from pydantic import ValidationError

from predictor import DiabetesPredictor, PatientInput

app = Flask(__name__)
predictor = DiabetesPredictor()


@app.route("/")
def index():
    return render_template("index.html", metrics=predictor.metrics)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"success": False, "error": "Invalid or missing JSON payload."}), 400

    try:
        patient_data = PatientInput(**payload)
        prediction_output = predictor.predict_all(patient_data)
        return jsonify({"success": True, "data": prediction_output})
    except ValidationError as exc:
        formatted_errors = [
            f"{err['loc'][0]}: {err['msg']}" for err in exc.errors()
        ]
        return jsonify({"success": False, "errors": formatted_errors}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)