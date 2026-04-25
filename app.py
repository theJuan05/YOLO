import os
import cv2
import time
import json
import traceback
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

# =========================
# STATS FILE
# =========================
STATS_FILE = "stats.json"

if not os.path.exists(STATS_FILE):
    with open(STATS_FILE, "w") as f:
        json.dump({"total_mask": 0, "total_no_mask": 0, "total_scans": 0}, f)

def load_stats():
    try:
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    except:
        return {"total_mask": 0, "total_no_mask": 0, "total_scans": 0}

def save_stats(mask, no_mask):
    stats = load_stats()
    stats["total_mask"] += mask
    stats["total_no_mask"] += no_mask
    stats["total_scans"] += 1

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)

    return stats


# =========================
# MODEL (LAZY LOAD)
# =========================
model = None

def get_model():
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), "best_final.pt")
        model = YOLO(model_path)
    return model
target_names = {
    0: "With Mask",
    1: "No Mask",
    2: "No Mask"
}


# =========================
# HOME ROUTE
# =========================
@app.route("/")
def home():
    stats = load_stats()
    return render_template("index.html", stats=stats)


# =========================
# PREDICT ROUTE (DEBUG FIXED)
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # -------------------------
        # CHECK FILE
        # -------------------------
        if "image" not in request.files:
            return "No file uploaded", 400

        file = request.files["image"]

        if file.filename == "":
            return "No file selected", 400

        os.makedirs("static", exist_ok=True)

        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        # -------------------------
        # IMAGE VALIDATION
        # -------------------------
        img = cv2.imread(filepath)
        if img is None:
            return "Invalid image upload", 400

        img = cv2.resize(img, (640, 640))
        cv2.imwrite(filepath, img)

        # -------------------------
        # YOLO MODEL
        # -------------------------
        model_instance = get_model()

        results = model_instance.predict(
            source=filepath,
            conf=0.4,
            imgsz=640,
            verbose=False
        )

        if results is None or len(results) == 0:
            return "YOLO failed", 500

        results[0].names = target_names

        # -------------------------
        # COUNT RESULTS
        # -------------------------
        mask_count = 0
        no_mask_count = 0

        if results[0].boxes is not None:
            for c in results[0].boxes.cls:
                if int(c) == 0:
                    mask_count += 1
                else:
                    no_mask_count += 1

        # -------------------------
        # SAVE STATS
        # -------------------------
        overall_stats = save_stats(mask_count, no_mask_count)

        # -------------------------
        # SAVE IMAGE
        # -------------------------
        annotated_frame = results[0].plot()
        if annotated_frame is not None:
            cv2.imwrite(filepath, annotated_frame)

        return render_template(
            "index.html",
            image=file.filename,
            mask=mask_count,
            no_mask=no_mask_count,
            stats=overall_stats,
            v=time.time()
        )

    except Exception as e:
        print(traceback.format_exc())  # THIS SHOWS REAL ERROR IN RENDER LOGS
        return f"SERVER ERROR: {str(e)}", 500


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)