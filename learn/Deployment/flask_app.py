import os
import logging
import joblib
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_PATH = os.getenv("MODEL_PATH", "/opt/ml/model/model.pkl")

app = Flask(__name__)
model = None  

def load_model(path: str):
    """Load model from file."""
    global model
    if not os.path.exists(path):
        logging.error(f"Model file not found at: {path}")
        raise FileNotFoundError(f"Model file not found at: {path}")
    #model = joblib.load(path)
    # a dummy model
    model = lambda x: 1
    logging.info(f"Model loaded from: {path}")


@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint."""
    health = model is not None
    status = 200 if health else 500
    return jsonify({"status": "ok" if health else "error"}), status


@app.route("/invocations", methods=["POST"])
def invocations():
    """Prediction endpoint."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid or missing JSON"}), 400

        # Example: Ensure input is a list of features
        if not isinstance(data, list):
            return jsonify({"error": "Input must be a list"}), 400

        prediction = model.predict([data])  # wrap in list for 2D
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    try:
        load_model(MODEL_PATH)
    except Exception as e:
        logging.exception("Model failed to load, exiting.")
        raise SystemExit(1)

    # Dev server — for production, use Gunicorn
    app.run(host="0.0.0.0", port=8080)
