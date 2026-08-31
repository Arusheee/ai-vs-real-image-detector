"""
Flask backend for the Real/Synthetic image scanner frontend.

Run:
    pip install flask flask-cors torch timm pillow
    python app.py

Then open index.html (served from this same folder, or via a simple
static server) — it calls this backend at http://localhost:5000/predict
"""

import os
import io
import gc

# Keep torch's internal thread pools small — on a 512MB instance, extra
# OpenMP/MKL threads each carry their own memory overhead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)

import timm
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms as T
from PIL import Image

MODEL_PATH = "best_modelNew.pt"
MODEL_ARCHITECTURE = "resnet18"
IMG_SIZE = 224
# Use the threshold you got from calibrate_threshold.py on external-test-images
AI_THRESHOLD = 0.68

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
model = timm.create_model(MODEL_ARCHITECTURE, pretrained=False, num_classes=2)
state_dict = torch.load(MODEL_PATH, map_location=device)
if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(f"Model loaded on: {device}")

eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    tensor = eval_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    real_prob = probs[0].item()
    ai_prob = probs[1].item()
    is_real = ai_prob <= AI_THRESHOLD

    del tensor, outputs, probs, img
    gc.collect()

    return jsonify({
        "verdict": "real" if is_real else "synthetic",
        "ai_probability": round(ai_prob * 100, 2),
        "real_probability": round(real_prob * 100, 2),
        "confidence": round((real_prob if is_real else ai_prob) * 100, 2),
        "threshold_used": AI_THRESHOLD,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(device)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)