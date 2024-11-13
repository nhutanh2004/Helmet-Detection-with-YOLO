from flask import Flask, render_template, request, jsonify, send_file
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
current_model_path = '/root/.cache/kagglehub/models/gbaonr/best5/pyTorch/default/1/best5.pt'
# current_model_path = 'C:\Users\phamh\.cache\kagglehub\models\gbaonr\best5\pyTorch\default\1\best5.pt'

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

    # Save the processed image to a file
    img_save_path = os.path.join('static', filename)
    img_with_detections = Image.fromarray(img_array)
    img_with_detections.save(img_save_path)

    return jsonify({"image_url": f"/static/{filename}"})

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
    output_path = os.path.join('static', f"processed_{filename}")
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
        results = model.predict(img_array)

        # Process YOLO results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if conf >= 0.6:  # Use the selected confidence threshold
                    label_text = f'{class_names[cls]}: {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Write the processed frame to the output video
        output.write((frame))

    cap.release()
    output.release()
    return jsonify({"video_url": f"/static/processed_{filename}"})

@app.route('/process_stream', methods=['POST'])
def process_stream():
    # Get the stream URL from the request
    stream_url = request.form.get('stream_url')
    if not stream_url:
        return jsonify({"error": "No stream URL provided"})
    
    # Connect to the network camera using the stream URL
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        return jsonify({"error": "Unable to connect to the stream"})

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for YOLO processing
        img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(img_array)

        # Process YOLO results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if conf >= 0.6:  # Use the selected confidence threshold
                    label_text = f'{class_names[cls]}: {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the frame in real-time
        cv2.imshow('Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "Stream processing completed"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

