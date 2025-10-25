from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import glob
import cv2
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = 'model/best.pt'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

def is_valid_image(filepath):
    try:
        img = cv2.imread(filepath)
        return img is not None
    except:
        return False

def is_pcb_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green_ratio = green_mask.sum() / (img.shape[0] * img.shape[1] * 255)
    return green_ratio > 0.05

def get_latest_result_image(after_time):
    matches = glob.glob(os.path.join(RESULT_FOLDER, '*.jpg'))
    matches = [f for f in matches if os.path.getmtime(f) > after_time]
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return os.path.basename(matches[0])

@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None
    error = None
    detection_message = None
    detections = []

    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            error = "No file selected."
            return render_template('index.html', error=error)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        if not is_valid_image(filepath):
            error = "Unsupported or corrupted image format."
            return render_template('index.html', error=error)

        if not is_pcb_image(filepath):
            error = "No PCB detected in the image."
            return render_template('index.html', error=error)

        detection_message = "🔍 Detection in progress... Please wait."
        start_time = time.time()

        try:
            results = model.predict(
                source=filepath,
                save=True,
                save_txt=False,
                save_conf=False,
                project='static',
                name='results',
                exist_ok=True,
                conf=0.25
            )
        except Exception as e:
            error = f"Prediction failed: {str(e)}"
            return render_template('index.html', error=error)

        result_image = get_latest_result_image(start_time)
        if not result_image:
            error = "Could not find result image."
        else:
            # Extract labels and confidence scores
            for r in results:
                for box in r.boxes:
                    label = model.names[int(box.cls)]
                    conf = float(box.conf)
                    detections.append((label, conf))

        detection_message = None  # Clear message after prediction

    return render_template('index.html',
                           result_image=result_image,
                           error=error,
                           detection_message=detection_message,
                           detections=detections)

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    app.run(debug=True)
