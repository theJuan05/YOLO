import os
import cv2
import time
import json
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

# File to store our total counts
STATS_FILE = "stats.json"

def load_stats():
    if not os.path.exists(STATS_FILE):
        return {"total_mask": 0, "total_no_mask": 0, "total_scans": 0}
    with open(STATS_FILE, "r") as f:
        return json.load(f)

def save_stats(mask, no_mask):
    stats = load_stats()
    stats["total_mask"] += mask
    stats["total_no_mask"] += no_mask
    stats["total_scans"] += 1
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)
    return stats

# 1. Load Model & Fix Metadata
model = YOLO("best_final.pt")
target_names = {0: 'With Mask', 1: 'No Mask', 2: 'No Mask'}

# Update model internal dictionaries to avoid KeyError: 2
if hasattr(model, 'model') and hasattr(model.model, 'names'):
    model.model.names = target_names
model.names.update(target_names)

@app.route("/")
def home():
    stats = load_stats() 
    return render_template("index.html", stats=stats)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files: return "No file", 400
    file = request.files["image"]
    
    if file.filename == '': return "No selection", 400

    if not os.path.exists('static'): os.makedirs('static')
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # 2. Run detection (verbose=False prevents terminal clutter/crashes)
    results = model.predict(source=filepath, conf=0.4, verbose=False)
    results[0].names = target_names

    mask_count = 0
    no_mask_count = 0
    
    if results[0].boxes is not None:
        for c in results[0].boxes.cls:
            class_id = int(c)
            # 0=With Mask, 1 & 2=No Mask/Incorrect
            if class_id == 0: 
                mask_count += 1
            else: 
                no_mask_count += 1

    # 3. Save to storage
    overall_stats = save_stats(mask_count, no_mask_count)

    # 4. Save result image
    annotated_frame = results[0].plot() 
    cv2.imwrite(filepath, annotated_frame) 

    return render_template("index.html", 
                           image=file.filename, 
                           mask=mask_count, 
                           no_mask=no_mask_count,
                           stats=overall_stats, 
                           v=time.time())

if __name__ == "__main__":
    # CRITICAL: use_reloader=False stops the infinite restart loop on Windows
    app.run(debug=True, use_reloader=False)