print("🚗 Setting up...")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

# Initialize Socket.IO server and Flask app
sio = socketio.Server()
app = Flask(__name__)
maxSpeed = 10

# Load trained model
print("📦 Loading model...")
model = load_model("carmodel/model.h5")
print("✅ Model loaded successfully!")


# Image preprocessing (NVIDIA model format)
def preProcess(img):
    img = img[60:135, :, :]  # Crop sky and car hood
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img


# Receive telemetry data from simulator
@sio.on("telemetry")
def telemetry(sid, data):
    print("📸 Telemetry received")
    print(data["Center"])
    # Current speed
    speed = float(data["Speed"])

    # Convert image from base64 to RGB numpy array
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image_array = np.asarray(image)
    image_array = preProcess(image_array)
    image_array = np.array([image_array])  # Add batch dimension

    # Predict steering angle
    try:
        pred = model.predict(image_array)
        print("Model raw output:", pred)
        steering = float(pred[0][0]) if pred.shape[-1] == 1 else float(pred[0])
    except Exception as e:
        print("Prediction error:", e)
        steering = 0.0

    # Calculate throttle based on speed
    throttle = max(0.0, min(1.0, 1.0 - speed / maxSpeed))

    # Show values
    print(f"➡️ Steering: {steering:.3f}, Throttle: {throttle:.3f}, Speed: {speed:.2f}")

    # Send control back to simulator
    sendControl(steering, throttle)


@sio.on("connect")
def connect(sid, environ):
    print("🔗 Simulator connected")
    sendControl(0, 0)  # Stop on connect


# Send steering & throttle back
def sendControl(steering, throttle):
    print("Sending control →", steering, throttle)
    sio.emit("steer", data={"steering_angle": str(steering), "throttle": str(throttle)})


# 🖥️ Launch server
if __name__ == "__main__":
    print("Starting server on port 4567...")
    app = socketio.WSGIApp(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
