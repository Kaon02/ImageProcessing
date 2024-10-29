import cv2
import numpy as np
import streamlit as st

# Function to process a frame to detect circles and annotate with score
def process_frame(frame):
    # Convert to RGB for drawing later
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge detection with Canny
    detected_edges = cv2.Canny(gray_img, 20, 210, 3)

    # Apply morphological closing to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close = cv2.morphologyEx(detected_edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(close, cv2.HOUGH_GRADIENT, 1.0, 25, param1=50, param2=20, minRadius=1, maxRadius=70)

    # Check if circles are detected
    if circles is not None:
        circles = circles[0, :]
        # Draw detected circles on the RGB frame
        for i in circles:
            # Draw the outer circle
            cv2.circle(rgb_frame, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(rgb_frame, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)

        # Put the score (number of detected circles) on the frame
        score_position = (30, 50)
        cv2.putText(rgb_frame, f'Score: {len(circles)}', score_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return rgb_frame

# Streamlit setup
st.title("Live Dice Detection with Streamlit")

# Instructions
st.write("This app detects dices in a live video stream and displays the score.")

# Access webcam using OpenCV
cap = cv2.VideoCapture(0)  # Change to 0 if using default webcam

if not cap.isOpened():
    st.error("Error: Could not open video stream from camera.")
else:
    # Streamlit real-time video frame display
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        # Process the frame to detect circles
        processed_frame = process_frame(frame)

        # Display the processed frame in Streamlit
        frame_placeholder.image(processed_frame, channels="RGB")

    # Release video capture and clean up
    cap.release()
