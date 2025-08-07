#!/usr/bin/env python3
import json
import os

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

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        alleles = data.get("alleles")
        peptides = data.get("peptides")

        if not alleles or not peptides:
            return jsonify({"error": "Both 'alleles' and 'peptides' are required"}), 400

        if len(alleles) != len(peptides) and not (
            len(alleles) == 1 or len(peptides) == 1
        ):
            return jsonify(
                {
                    "error": "Alleles and peptides must have same length or one of them must be singular"
                }
            ), 400

        if len(alleles) == 1 and len(peptides) == 1:
            predictions_df = predictor.predict_to_dataframe(
                peptides=peptides, allele=alleles[0]
            )
        else:
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
