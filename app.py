from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
import os
import io
import cv2
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from ensemble_boxes import weighted_boxes_fusion
import torch

app = Flask(__name__)

# Ensure static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Ensure weights directory exists
if not os.path.exists('weights'):
    os.makedirs('weights')

# Load all YOLO models from the weights directory
def load_models():
    models = []
    weights_dir = 'weights'
    for model_name in os.listdir(weights_dir):
        model_path = os.path.join(weights_dir, model_name)
        if model_path.endswith('.pt'):
            models.append(YOLO(model_path, task='detect'))
    return models

# Load models
models = load_models()

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

    all_boxes, all_scores, all_labels = [], [], []

    # Make predictions using all models
    for model in models:
        results = model.predict(img_array)
        boxes, labels, scores = [], [], []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                if conf >= confidence_threshold:
                    boxes.append([x1 / img_resized.width, y1 / img_resized.height, x2 / img_resized.width, y2 / img_resized.height])
                    labels.append(cls)
                    scores.append(conf)
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    # Apply Weighted Box Fusion
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(all_boxes, all_scores, all_labels, iou_thr=0.5, skip_box_thr=0.001)

    # Draw the bounding boxes
    for i in range(len(fused_boxes)):
        x1, y1, x2, y2 = map(int, [fused_boxes[i][0] * img_resized.width, fused_boxes[i][1] * img_resized.height, fused_boxes[i][2] * img_resized.width, fused_boxes[i][3] * img_resized.height])
        label = int(fused_labels[i])  # Ensure label is an integer
        score = fused_scores[i]
        label_text = f'{class_names[label]}: {score:.2f}'
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_array, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save the processed image to a file with a new name
    img_save_path = os.path.join('static', f'predicted_{filename}')
    img_with_detections = Image.fromarray(img_array)
    img_with_detections.save(img_save_path)

    return jsonify({"image_url": f"/static/predicted_{filename}"})

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    filename = secure_filename(file.filename)
    file_path = os.path.join('static', filename)
    file.save(file_path)
    
    # Open the video file
    cap = cv2.VideoCapture(file_path)
    output_path = os.path.join('static', f"predicted_{filename}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_path, cv2_fourcc , 30, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for YOLO processing
        img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        all_boxes, all_scores, all_labels = [], [], []

        # Make predictions using all models
        for model in models:
            results = model.predict(img_array)
            boxes, labels, scores = [], [], []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    if conf >= 0.6:  # Use the selected confidence threshold
                        boxes.append([x1 / frame_width, y1 / frame_height, x2 / frame_width, y2 / frame_height])
                        labels.append(cls)
                        scores.append(conf)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        # Apply Weighted Box Fusion
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(all_boxes, all_scores, all_labels, iou_thr=0.5, skip_box_thr=0.001)

        # Draw the bounding boxes
        for i in range(len(fused_boxes)):
            x1, y1, x2, y2 = map(int, [fused_boxes[i][0] * frame_width, fused_boxes[i][1] * frame_height, fused_boxes[i][2] * frame_width, fused_boxes[i][3] * frame_height])
            label = int(fused_labels[i])  # Ensure label is an integer
            score = fused_scores[i]
            label_text = f'{class_names[label]}: {score:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Write the processed frame to the output video
        output.write(frame)

    cap.release()
    output.release()
    return jsonify({"video_url": f"/static/predicted_{filename}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


