from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# --- LOAD MODEL ---
# Loads once when the server starts (equivalent to @st.cache_resource)
model = YOLO('best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read image into OpenCV format
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Inference (Exactly as Streamlit logic)
    results = model.predict(source=image, conf=0.45)
    res_plotted = results[0].plot(line_width=3)
    
    # Encode plotted image to base64 so HTML can display it
    # Note: OpenCV naturally encodes BGR to proper JPG format, so BGR2RGB conversion is not needed here
    _, buffer = cv2.imencode('.jpg', res_plotted)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    # Status / Box Extraction logic
    num_boxes = len(results[0].boxes)
    detections = []
    
    if num_boxes > 0:
        for box in results[0].boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            detections.append({'label': label, 'conf': conf})

    return jsonify({
        'image': encoded_img,
        'num_boxes': num_boxes,
        'detections': detections
    })

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)
