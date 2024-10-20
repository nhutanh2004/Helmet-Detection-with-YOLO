from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import kagglehub
import os
import io
import cv2
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ensure static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Path to the model
current_model_path = r'C:\Users\phamh\.cache\kagglehub\models\gbaonr\best5\pyTorch\default\1\best5.pt'

# Function to check if the model needs to be updated
def update_model():
    if not os.path.exists(current_model_path):
        print("Model not found. Downloading...")
        kagglehub.model_download("gbaonr/best5/pyTorch/default")
    else:
        print("Model is up-to-date.")
    return current_model_path

# Load the YOLO model
model_path = update_model()
model = YOLO(model_path, task='detect')

# Define class names
class_names = ['motorbike', 'DHelmet', 'DNoHelmet', 'P1Helmet', 'P1NoHelmet', 'P2Helmet', 'P2NoHelmet', 'P0Helmet', 'P0NoHelmet']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    filename = secure_filename(file.filename)
    confidence_threshold = float(request.form.get('confidenceThreshold', 0.6))

    # Open image
    img = Image.open(io.BytesIO(file.read()))
    img = img.convert("RGB")
    img_resized = img.resize((640, 640))
    img_array = np.array(img_resized)

    # Make predictions
    results = model.predict(img_array)

    if not results:
        print("No results found")
    
    boxes = []
    labels = []
    scores = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if conf >= confidence_threshold:  # Use the selected confidence threshold
                boxes.append([x1, y1, x2, y2])
                labels.append(cls)
                scores.append(conf)
                # Draw the bounding box
                cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f'{class_names[cls]}: {conf:.2f}'
                cv2.putText(img_array, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Debugging predictions
    print(f"Boxes: {boxes}")
    print(f"Labels: {labels}")
    print(f"Scores: {scores}")

    # Save the image with drawn boxes directly without converting color space
    img_save_path = os.path.join('static', filename)
    img_to_save = Image.fromarray(img_array)  # Convert back to Image format if needed
    img_to_save.save(img_save_path)

    return jsonify({
        "filename": filename,
        "boxes": boxes,
        "labels": labels,
        "scores": scores,
        "image_url": f"static/{filename}"
    })

if __name__ == '__main__':
    app.run(debug=True)
