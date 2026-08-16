import os
import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
from io import BytesIO

# Load YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train23', 'weights', 'last.pt')
model = YOLO(model_path)

# Streamlit Interface
st.title("Object Detection with Native Video Player")

# File upload for images or videos
uploaded_file = st.file_uploader("Select an Image or Video file for Object Detection", type=["mp4", "avi", "mov", "jpg", "jpeg", "png"])

# Threshold for detection confidence
threshold = st.slider("Set Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Function to process and display image with object detection
def process_image(image_bytes, model, threshold):
    # Read image from bytes
    image = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Perform object detection
    results = model(image)[0]

    # Draw bounding boxes on the image (Red for filled, Green for empty)
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            color = (0, 255, 0) if results.names[int(class_id)].lower() == 'empty' else (0, 0, 255)  # Red for filled, Green for empty
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Smaller thickness for cleaner look
            cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)  # Smaller font size and thickness

    # Display the processed image
    st.image(image, channels="BGR")

# Function to process and display video with object detection and controls
def process_video(video_bytes, model, threshold):
    # Save the uploaded video to a temporary file
    video_file = BytesIO(video_bytes)  # Convert video bytes to BytesIO
    video_path = './uploaded_video.mp4'
    with open(video_path, 'wb') as f:
        f.write(video_file.read())  # Write the video data to the temporary file

    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open the video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Setup for Streamlit video player window
    video_bytes = open(video_path, 'rb').read()  # Read the video file as bytes
    st.video(video_bytes, start_time=0)

    # Process video with object detection
    frame_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Perform object detection on the frame
        results = model(frame)[0]

        # Draw bounding boxes on the frame (Red for filled, Green for empty)
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                color = (0, 255, 0) if results.names[int(class_id)].lower() == 'empty' else (0, 0, 255)  # Red for filled, Green for empty
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Smaller thickness for cleaner look
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)  # Smaller font size and thickness

        # Display the processed frame
        frame_placeholder.image(frame, channels="BGR")
        # use_container_width=True
    cap.release()

# Main logic for handling uploaded files
if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    if file_extension in ['.mp4', '.avi', '.mov']:
        st.write("Processing video...")
        process_video(uploaded_file.read(), model, threshold)
    elif file_extension in ['.jpg', '.jpeg', '.png']:
        st.write("Processing image...")
        process_image(uploaded_file.read(), model, threshold)
    else:
        st.error("Unsupported file type. Please upload a valid image or video.")

