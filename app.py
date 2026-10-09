# imports
import os
from flask import Flask, jsonify, render_template, request

from src.config import FEATURES
from src.predict import CardiacRiskPredictor


# init
a = Flask(__name__)
app = a
b = CardiacRiskPredictor()


# groups
def gf() -> dict:
    g = {}
    for f in FEATURES:
        g.setdefault(f["group"], []).append(f)
    return g


# home
@a.get("/")
def idx():
    return render_template("index.html", groups=gf(), metrics=b.metrics)




# predict
@a.post("/api/predict")
def pred():
    if not request.is_json and request.content_length:
        return jsonify({"error": "Request content-type must be application/json"}), 415
    try:
        p = request.get_json(silent=False)
        if p is None:
            p = {}
        elif not isinstance(p, dict):
            return jsonify({"error": "JSON payload must be a JSON object (key-value mapping)."}), 400
    except Exception:
        return jsonify({"error": "Malformed JSON payload in request."}), 400

    try:
        r = b.predict(p)
        return jsonify(r), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Internal prediction engine failure: {str(e)}"}), 500


# metrics
@a.get("/api/metrics")
def met():
    try:
        return jsonify(b.metrics), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve model metrics: {str(e)}"}), 500


# health
@a.get("/health")
def hlt():
    m = bool(b and b.models)
    s = 200 if m else 503
    return jsonify({
        "status": "healthy" if m else "degraded",
        "models_loaded": len(b.models) if m else 0
    }), s


# missing
@a.errorhandler(404)
def e404(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found."}), 404
    return render_template("index.html", groups=gf(), metrics=b.metrics), 404


# not allowed
@a.errorhandler(405)
def e405(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "HTTP method not allowed for this endpoint."}), 405
    return jsonify({"error": "Method not allowed."}), 405


# failure
@a.errorhandler(500)
def e500(e):
    return jsonify({"error": "Internal server error."}), 500


# entry
if __name__ == "__main__":
    a.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
          debug=os.environ.get("FLASK_DEBUG") == "1")
