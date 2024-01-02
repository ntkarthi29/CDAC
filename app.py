import streamlit as st
from ultralytics import YOLO
import cv2

st.title("YOLO Object Detection")

# Load the YOLO model
model = YOLO("./yolov8_custom.pt")

video_capture = None
stop_after_n_images = 10
image_counter = 0

def start_webcam():
    global video_capture
    video_capture = cv2.VideoCapture(0)

def stop_webcam():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None

def process_frame(frame):
    global image_counter
    # Convert frame to RGB (YOLO expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(frame_rgb)

    # Iterate through the detection results and draw bounding boxes manually
    for result in results:
        boxes = result.boxes.data.tolist()
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            class_idx = int(box[-1])  # Retrieve the class index
            confidence = box[-2]  # Retrieve the confidence

            # Add text to the image
            class_name = model.names[class_idx]
            text = f"{class_name}: {confidence:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness=1)

    # Convert back to BGR for display
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def process_video_frame():
    global image_counter
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            st.error("Error reading frame from webcam")
            break  # Exit the loop if frame reading fails

        frame_with_boxes = process_frame(frame)

        yield frame_with_boxes

        # Increment image counter
        image_counter += 1

        # Stop after a certain number of images
        if image_counter >= stop_after_n_images:
            stop_webcam()
            break

if st.button("Start Webcam"):
    start_webcam()

if st.button("Stop Webcam"):
    stop_webcam()
    st.success("Webcam stopped")

if video_capture is not None:
    for frame in process_video_frame():
        frame = cv2.imencode('.jpg', frame)[1].tobytes()  # Encode as JPEG bytes
        st.image(frame, channels="BGR")  # Display using st.image

    # Ask the user if they want to detect more images
    more_images = st.checkbox("Detect more images?")
    if more_images:
        start_webcam()
