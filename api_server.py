#!/usr/bin/env python3
import json

from flask import Flask, jsonify, request

from mhcflurry import Class1AffinityPredictor

app = Flask(__name__)

predictor = Class1AffinityPredictor.load()


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "mhcflurry API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or not isinstance(data, list):
            return jsonify(
                {
                    "error": "JSON data must be a list of {'allele': ..., 'peptide': ...} dictionaries."
                }
            ), 400

        if not all("allele" in d and "peptide" in d for d in data):
            return jsonify(
                {
                    "error": "Each item in the list must be a dictionary with 'allele' and 'peptide' keys"
                }
            ), 400

        peptides = [d["peptide"] for d in data]
        alleles = [d["allele"] for d in data]

        predictions_df = predictor.predict_to_dataframe(
            peptides=peptides, alleles=alleles
        )

        result = []
        for i in range(len(predictions_df)):
            row = predictions_df.iloc[i]
            result.append(
                {
                    "allele": row["allele"],
                    "peptide": row["peptide"],
                    "affinity": float(row["prediction"]),
                    "percentile": float(row["prediction_percentile"]),
                }
            )

        return jsonify({"predictions": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/supported_alleles", methods=["GET"])
def get_supported_alleles():
    try:
        return jsonify({"supported_alleles": predictor.supported_alleles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
