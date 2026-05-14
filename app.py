import os
import io
import cv2
import numpy as np
from flask import Flask, request, send_file
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Load the model once when the app starts
model = YOLO('best.pt')

# 1. Provide a simple HTML upload page
@app.route('/', methods=['GET'])
def index():
    return '''
    <!doctype html>
    <html lang="en">
    <head>
      <title>Plant Disease Detection</title>
      <style>
        body { font-family: Arial, sans-serif; background-color: #f4f9f4; text-align: center; padding-top: 50px; }
        .container { background: white; padding: 30px; border-radius: 12px; display: inline-block; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h2 { color: #2e7d32; }
        input[type="submit"] { background-color: #4caf50; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
      </style>
    </head>
    <body>
      <div class="container">
        <h2>🌿 Plant Disease Detector</h2>
        <p>Upload a suspected plant leaf image.</p>
        <form method="post" action="/predict" enctype="multipart/form-data">
          <input type="file" name="file" accept=".jpg, .jpeg, .png" required>
          <br><br>
          <input type="submit" value="Analyze Leaf">
        </form>
      </div>
    </body>
    </html>
    '''

# 2. Handle the image upload and run YOLO inference
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    if file:
        # Read the image
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run YOLO prediction
        results = model.predict(source=img, conf=0.45)
        
        # Plot the bounding boxes
        res_plotted = results[0].plot(line_width=3)
        
        # Convert BGR (OpenCV format) to RGB (PIL format)
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        res_img = Image.fromarray(res_rgb)

        # Save the result to memory to send back to the browser
        img_io = io.BytesIO()
        res_img.save(img_io, 'JPEG', quality=90)
        img_io.seek(0)

        # Send the image to the user
        return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    # Bind to the port provided by Render, or default to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)