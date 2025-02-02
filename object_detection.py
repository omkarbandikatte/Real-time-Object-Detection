import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

@st.cache_resource(show_spinner=False)
def load_model():
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")
    return net

net = load_model()

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def detect_objects(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if idx < len(CLASSES):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = f"{CLASSES[idx]}: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, max(startY - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame

st.title("Real-Time Person Detection with MobileNet SSD")

if "run" not in st.session_state:
    st.session_state.run = False
if "captured" not in st.session_state:
    st.session_state.captured = False

st.sidebar.header("Camera Control")
run_checkbox = st.sidebar.checkbox("Run Camera", value=st.session_state.run)
if run_checkbox != st.session_state.run:
    st.session_state.run = run_checkbox
    if not run_checkbox:
        st.session_state.captured = False

capture_image = st.sidebar.button("Capture Image")

uploaded_file = st.sidebar.file_uploader("Or Upload an Image", type=["jpg", "jpeg", "png"])

FRAME_WINDOW = st.image([])

if st.session_state.run:

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam. Make sure it is connected and not used by another application.")
    else:
        st.info("Press the sidebar 'Capture Image' button to save a snapshot.")
    while st.session_state.run and cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.error("Failed to capture video frame", icon="🚨")
            break

        frame = detect_objects(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

        if capture_image and not st.session_state.captured:
            img = Image.fromarray(frame_rgb)
            img.save("captured_image.jpg")
            st.success("Image Captured and saved as captured_image.jpg")
            st.session_state.captured = True

        time.sleep(0.05)

    cap.release()

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is not None:
        image = detect_objects(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Uploaded Image with Detections")
    else:
        st.error("Error processing uploaded image.")

st.markdown("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Object Detection</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
</head>
<body>
    <div class="background"></div>
    <div class="container">
        <h1 class="title">Real-time Person Detection</h1>
        <div class="video-container">
            <!-- The video feed is provided by the Streamlit app above -->
            <div id="loading" style="display: none;">Loading...</div>
        </div>
        <p class="instructions">AI-powered detection in real-time. Use the sidebar controls to operate the camera.</p>
    </div>
</body>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');
    body {
        margin: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        background: url('/static/background.jpg') no-repeat center center/cover;
    }
    .background {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.6);
    }
    .container {
        position: relative;
        text-align: center;
        padding: 20px;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
    }
    .title {
        color: #ffcc00;
        font-size: 32px;
        font-family: 'Poppins', sans-serif;
        font-weight: bold;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
        margin-bottom: 15px;
    }
    .video-container {
        position: relative;
    }
    #loading {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 20px;
        color: white;
        background: rgba(0, 0, 0, 0.6);
        padding: 10px 20px;
        border-radius: 5px;
    }
    .instructions {
        color: white;
        font-size: 16px;
        margin-top: 10px;
    }
</style>
</html>
""", unsafe_allow_html=True)
