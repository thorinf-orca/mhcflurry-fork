#!/usr/bin/env python3
import logging
import os

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

VALID_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
VALID_AMINO_ACIDS_SET = set(VALID_AMINO_ACIDS)

# Configuration from environment variables
MAX_REQUEST_SIZE = int(
    os.getenv("MAX_REQUEST_SIZE", "10000")
)  # Max items in JSON request
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1024"))  # Max items processed at once

logger.info(
    f"Configuration: MAX_REQUEST_SIZE={MAX_REQUEST_SIZE}, MAX_BATCH_SIZE={MAX_BATCH_SIZE}"
)
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

        if len(data) > MAX_REQUEST_SIZE:
            logger.warning(
                f"Request too large: {len(data)} items (max {MAX_REQUEST_SIZE})"
            )
            return jsonify(
                {"error": f"Request too large (max {MAX_REQUEST_SIZE} items)"}
            ), 400

        # Process each item and collect valid ones for prediction
        valid_items = []
        response_items = []

        for i, item in enumerate(data):
            error = None

            if not isinstance(item, dict):
                error = "Item must be a dictionary"
            elif "allele" not in item or "peptide" not in item:
                error = "Item must contain 'allele' and 'peptide' keys"
            elif not isinstance(item["allele"], str) or not isinstance(
                item["peptide"], str
            ):
                error = "'allele' and 'peptide' must be strings"
            elif not item["allele"].strip() or not item["peptide"].strip():
                error = "'allele' and 'peptide' cannot be empty"
            else:
                allele = item["allele"].strip()
                peptide = item["peptide"].strip()

                if allele not in predictor.supported_alleles:
                    error = f"Unsupported allele: {allele}"
                elif not all(char in VALID_AMINO_ACIDS_SET for char in peptide.upper()):
                    invalid_chars = set(peptide.upper()) - VALID_AMINO_ACIDS_SET
                    error = f"Invalid amino acid characters: {', '.join(sorted(invalid_chars))}"
                else:
                    peptide_len = len(peptide)
                    if peptide_len < 5 or peptide_len > 15:
                        error = f"Peptide length {peptide_len} outside supported range [5, 15]"
                    else:
                        if peptide_len < 8:
                            logger.warning(
                                f"Unusual peptide length {peptide_len} for peptide: {peptide}"
                            )

                        valid_items.append(
                            {"index": i, "allele": allele, "peptide": peptide}
                        )

            if error:
                allele = item.get("allele", "N/A") if isinstance(item, dict) else "N/A"
                peptide = (
                    item.get("peptide", "N/A") if isinstance(item, dict) else "N/A"
                )
                response_items.append(
                    {"allele": allele, "peptide": peptide, "error": error}
                )
            else:
                response_items.append(None)  # Placeholder for prediction

        # Run predictions on valid items in chunks
        if valid_items:
            logger.info(
                f"Processing {len(valid_items)} predictions in chunks of {MAX_BATCH_SIZE}, {len(data) - len(valid_items)} invalid items"
            )

            # Process valid items in chunks
            for chunk_start in range(0, len(valid_items), MAX_BATCH_SIZE):
                chunk_end = min(chunk_start + MAX_BATCH_SIZE, len(valid_items))
                chunk_items = valid_items[chunk_start:chunk_end]

                peptides = [item["peptide"] for item in chunk_items]
                alleles = [item["allele"] for item in chunk_items]

                logger.info(
                    f"Processing chunk {chunk_start // MAX_BATCH_SIZE + 1}: {len(chunk_items)} items"
                )

                predictions_df = predictor.predict_to_dataframe(
                    peptides=peptides, alleles=alleles
                )

                # Fill in predictions for this chunk
                for i, valid_item in enumerate(chunk_items):
                    row = predictions_df.iloc[i]
                    response_items[valid_item["index"]] = {
                        "allele": row["allele"],
                        "peptide": row["peptide"],
                        "affinity": float(row["prediction"]),
                        "percentile": float(row["prediction_percentile"]),
                    }
        else:
            logger.info(f"No valid items to process, all {len(data)} items invalid")

        valid_count = len(valid_items)
        total_count = len(data)
        filtered_count = total_count - valid_count

        return jsonify(
            {
                "predictions": response_items,
                "valid_count": valid_count,
                "filtered_count": filtered_count,
                "total_count": total_count,
            }
        )

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
