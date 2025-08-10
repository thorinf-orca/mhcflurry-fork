#!/usr/bin/env python3
import logging

import tensorflow as tf
from flask import Flask, jsonify, request

from mhcflurry import Class1AffinityPredictor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def configure_tensorflow():
    try:
        physical_devices = tf.config.list_physical_devices()
        logger.info(f"Available devices: {physical_devices}")

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured {len(gpus)} GPU(s) with memory growth")
            except RuntimeError as e:
                logger.error(f"GPU configuration failed: {e}")

        if tf.config.list_physical_devices("GPU"):
            logger.info("GPU devices found - will attempt to use for acceleration")

        if gpus or any("GPU" in str(device) for device in physical_devices):
            try:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Enabled mixed precision for better GPU performance")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")

        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"GPU available: {tf.config.list_physical_devices('GPU')}")
        logger.info(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

    except Exception as e:
        logger.error(f"TensorFlow configuration failed: {e}")
        logger.info("Falling back to CPU")


configure_tensorflow()

app = Flask(__name__)

logger.info("Loading MHCflurry predictor...")
predictor = Class1AffinityPredictor.load()
logger.info("MHCflurry predictor loaded successfully")


@app.route("/health", methods=["GET"])
def health_check():
    try:
        device_info = {
            "tensorflow_version": tf.__version__,
            "gpu_available": len(tf.config.list_physical_devices("GPU")) > 0,
            "physical_devices": [str(d) for d in tf.config.list_physical_devices()],
            "cuda_built": tf.test.is_built_with_cuda()
            if hasattr(tf.test, "is_built_with_cuda")
            else False,
        }
        return jsonify(
            {
                "status": "healthy",
                "message": "mhcflurry API is running",
                "device_info": device_info,
                "supported_alleles_count": len(predictor.supported_alleles),
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "degraded", "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            logger.warning("Request without JSON content-type")
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()
        if not data:
            logger.warning("Empty JSON data received")
            return jsonify({"error": "Empty JSON data"}), 400

        if not isinstance(data, list):
            logger.warning(f"Invalid data type: {type(data)}")
            return jsonify(
                {
                    "error": "JSON data must be a list of {'allele': ..., 'peptide': ...} dictionaries."
                }
            ), 400

        if len(data) == 0:
            logger.warning("Empty list received")
            return jsonify({"error": "List cannot be empty"}), 400

        if len(data) > 1000:
            logger.warning(f"Large batch size: {len(data)}")
            return jsonify({"error": "Batch size too large (max 1000)"}), 400

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                return jsonify({"error": f"Item {i} must be a dictionary"}), 400

            if "allele" not in item or "peptide" not in item:
                return jsonify(
                    {"error": f"Item {i} must contain 'allele' and 'peptide' keys"}
                ), 400

            if not isinstance(item["allele"], str) or not isinstance(
                item["peptide"], str
            ):
                return jsonify(
                    {"error": f"Item {i}: 'allele' and 'peptide' must be strings"}
                ), 400

            if not item["allele"].strip() or not item["peptide"].strip():
                return jsonify(
                    {"error": f"Item {i}: 'allele' and 'peptide' cannot be empty"}
                ), 400

            peptide_len = len(item["peptide"].strip())
            if peptide_len < 8 or peptide_len > 15:
                logger.warning(
                    f"Unusual peptide length {peptide_len} for peptide: {item['peptide']}"
                )

        peptides = [d["peptide"].strip() for d in data]
        alleles = [d["allele"].strip() for d in data]

        logger.info(f"Processing {len(peptides)} predictions")

        unsupported_alleles = set(alleles) - set(predictor.supported_alleles)
        if unsupported_alleles:
            return jsonify(
                {
                    "error": f"Unsupported alleles: {list(unsupported_alleles)}",
                    "supported_alleles_count": len(predictor.supported_alleles),
                }
            ), 400

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

        logger.info(f"Successfully processed {len(result)} predictions")
        return jsonify({"predictions": result, "count": len(result)})

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"error": f"Validation error: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/supported_alleles", methods=["GET"])
def get_supported_alleles():
    try:
        return jsonify({"supported_alleles": predictor.supported_alleles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
