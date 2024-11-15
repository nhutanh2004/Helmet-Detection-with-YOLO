from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from ultralytics import YOLO
import os
import io
import cv2
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from ensemble_boxes import weighted_boxes_fusion
import torch
import ffmpeg
import subprocess

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

# @app.route('/')
# def home():
#     return render_template('index.html')
@app.route('/') 
def home(): 
    return render_template('home.html') 

@app.route('/image') 
def upload_image(): 
    return render_template('image.html') 

@app.route('/video') 
def upload_video():
    return render_template('video.html')



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
    if not cap.isOpened():
        return jsonify({"error": "Unable to open video file"})

    # Get frame dimensions and calculate target size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) # Ensure this matches the original frame rate
    # frame_rate = 24  
    target_width = 640
    aspect_ratio = frame_height / frame_width
    target_height = int(target_width * aspect_ratio)
    
    output_path = os.path.join('static', f"predicted_{filename}")
    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_path, cv2_fourcc, frame_rate, (target_width, target_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to the target size
        frame = cv2.resize(frame, (target_width, target_height))

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
                        boxes.append([x1 / target_width, y1 / target_height, x2 / target_width, y2 / target_height])
                        labels.append(cls)
                        scores.append(conf)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        # Apply Weighted Box Fusion
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(all_boxes, all_scores, all_labels, iou_thr=0.5, skip_box_thr=0.001)

        # Draw the bounding boxes
        for i in range(len(fused_boxes)):
            x1, y1, x2, y2 = map(int, [fused_boxes[i][0] * target_width, fused_boxes[i][1] * target_height, fused_boxes[i][2] * target_width, fused_boxes[i][3] * target_height])
            label = int(fused_labels[i])  # Ensure label is an integer
            score = fused_scores[i]
            label_text = f'{class_names[label]}: {score:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the processed frame in real-time (commented out for web use)
        # cv2.imshow('Processed Video', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # Write the processed frame to the output video
        output.write(frame)

    cap.release()
    output.release()
    #cv2.destroyAllWindows()

   # Compress the video using ffmpeg command line and capture stderr 
    compressed_output_path = os.path.join('static', f"compressed_{filename}") 
    ffmpeg_cmd = ['./ffmpeg.exe',
                  '-i', output_path, 
                  '-vcodec', 'h264', 
                  '-acodec', 'aac', 
                  '-strict', '-2', 
                  compressed_output_path 
                  ] 
    # Debug for some time the compressed_output does not save properly
    try:
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
        else:
            print(f"Compressed video created successfully at: {compressed_output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    # Delete the intermediate processed video file  
    if os.path.exists(output_path): 
       os.remove(output_path)
    
    # Delete the original video file  
    if os.path.exists(file_path): 
        os.remove(file_path)
        
   

    return jsonify({"video_url": f"/static/compressed_{filename}"})

@app.route('/download/<filename>', methods=['GET']) 
def download_file(filename): 
    return send_from_directory('static', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


