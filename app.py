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
from class_colors import class_names, class_colors # Import the class names and colors

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

# Function to count samples per class

def count_samples_per_class(data):
    class_counts = [0] * len(class_names)
    for line in data:
        #print(f"Line: {line}")  # Debug print
        if len(line) > 1:  # Ensure the line has enough elements
            class_id = int(line[-2])
            #print(f"Class ID: {class_id}")  # Debug print for class ID
            class_counts[class_id] += 1
            #print(f"Updated class counts: {class_counts}")  # Debug print for updated counts
        else:
            print(f"Skipped line due to insufficient length: {line}")  # Debug print for skipped lines
    return class_counts, max(class_counts)


# Function to find max
def find_max(data):
    classes_count, n_max_class = count_samples_per_class(data)
    return classes_count, n_max_class

# Function to find the minority class and threshold
def minority(p, data, n=9):
    classes_count, n_max_class = find_max(data)
    mean_samples = float(sum(classes_count) / n) if n > 0 else float('inf')  # mean samples per class
    alpha = float(n_max_class / mean_samples) if mean_samples != 0 else 0  # mean samples per class / max samples in a class

    rare_classes = set()

    # find rare classes
    for index, each_class in enumerate(classes_count):
        if each_class < (n_max_class * alpha):
            rare_classes.add(index)

    min_thresh = 1

    # find minimum threshold
    for each_class_index in rare_classes:
        for sample in data:
            if len(sample) > 1:  # Ensure there's an adequate number of elements
                class_id = sample[-2]
                score = sample[-1]

                if class_id != each_class_index:
                    continue
                if score < min_thresh:
                    min_thresh = score

    return max(min_thresh, p), rare_classes


# Function to optimize minority classes
def minority_optimizer_func(boxes, labels, scores, p=0.001):
    number_of_classes = len(class_names)
    minority_score, rare_classes = minority(p, list(zip(boxes, labels, scores)), number_of_classes)
    
    #print(f"Minority score: {minority_score}, Rare classes: {rare_classes}")
    
    optimized_boxes, optimized_labels, optimized_scores = [], [], []
    for i in range(len(scores)):
        if scores[i] >= minority_score:
            optimized_boxes.append(boxes[i])
            optimized_labels.append(labels[i])
            optimized_scores.append(scores[i])
    
    # print(f"Optimized boxes: {optimized_boxes}")
    # print(f"Optimized labels: {optimized_labels}")
    # print(f"Optimized scores: {optimized_scores}")
    
    return optimized_boxes, optimized_labels, optimized_scores


@app.route('/') 
def home(): 
    return render_template('video.html') 

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

    # Get parameters from the request
    iou_thr = float(request.form['iou_thr'])
    skip_box_thr = float(request.form['skip_box_thr'])
    p = float(request.form['p'])

    # Open the video file
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return jsonify({"error": "Unable to open video file"})

    # Get frame dimensions and calculate target size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    target_fps = 10  # Desired frames per second for processing
    if frame_rate < target_fps:
        target_fps = frame_rate  # Ensure target_fps is not greater than frame_rate
    frame_interval = frame_rate // target_fps  # Interval to skip frames

    target_width = 640
    aspect_ratio = frame_height / frame_width
    target_height = int(target_width * aspect_ratio)
    
    output_path = os.path.join('static', f"predicted_{filename}")
    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_path, cv2_fourcc, target_fps, (target_width, target_height))

    frame_counter = 0  # Initialize frame counter

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to meet the target FPS
        if frame_counter % frame_interval != 0:
            frame_counter += 1
            continue

        # Resize frame to the target size
        frame = cv2.resize(frame, (target_width, target_height))

        # Convert frame to RGB for YOLO processing
        img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        all_boxes, all_scores, all_labels = [], [], []

        # Use the batch size and device from the second function
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        confidence_threshold = 0.00001  # Use the confidence threshold from the second function

        # Make predictions using all models
        for model in models:
            results = model.predict(img_array, save=False, stream=True, batch=8, conf=confidence_threshold, device=device)
            boxes, labels, scores = [], [], []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    if conf >= confidence_threshold:  # Use the selected confidence threshold
                        boxes.append([x1 / target_width, y1 / target_height, x2 / target_width, y2 / target_height])
                        labels.append(cls)
                        scores.append(conf)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        # Apply Weighted Box Fusion with user parameters
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(all_boxes, all_scores, all_labels, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

        # Optimize minority classes
        optimized_boxes, optimized_labels, optimized_scores = minority_optimizer_func(fused_boxes, fused_labels, fused_scores,p=p)

        # Draw the bounding boxes with specific colors
        for i in range(len(optimized_boxes)):
            x1, y1, x2, y2 = map(int, [optimized_boxes[i][0] * target_width, optimized_boxes[i][1] * target_height, optimized_boxes[i][2] * target_width, optimized_boxes[i][3] * target_height])
            label = int(optimized_labels[i])  # Ensure label is an integer
            score = optimized_scores[i]
            color_rgb = class_colors[label]  # Get the RGB color for the class
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # Convert RGB to BGR
            label_text = f'{class_names[label]}: {score:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

        # Write the processed frame to the output video
        output.write(frame)
        frame_counter += 1  # Increment frame counter

    cap.release()
    output.release()

    # Compress the processed video using ffmpeg command line and capture stderr
    compressed_output_path = os.path.join('static', f"compressed_{filename}")
    ffmpeg_cmd = ['./ffmpeg.exe',
                  '-i', output_path,
                  '-vcodec', 'h264',
                  '-acodec', 'aac',
                  '-strict', '-2',
                  '-r', str(target_fps),  # Ensure the frame rate matches target_fps during compression
                  compressed_output_path
                  ]
    try:
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
        else:
            print(f"Compressed processed video created successfully at: {compressed_output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Delete the intermediate processed video file
    if os.path.exists(output_path):
        os.remove(output_path)
    
    return jsonify({"original_video_url": f"/static/{filename}", "processed_video_url": f"/static/compressed_{filename}"})

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory('static', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


