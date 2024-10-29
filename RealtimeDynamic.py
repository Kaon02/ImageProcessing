import streamlit as st
import cv2
import numpy as np

# Define stable colors for contours
stable_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

# Define the processing function
def process_liveDynamic(frame):
    max_width = 800
    height, width = frame.shape[:2]
    if width > max_width:
        scaling_factor = max_width / width
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        frame = cv2.resize(frame, new_size)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(thresh, 50, 200)

    # Use morphological transformations to close small gaps in contours
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    size_groups = []
    min_area_threshold = 100  

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area_threshold:
            continue

        # Check if this area fits into an existing size group
        found_group = False
        for idx, (group_area, count) in enumerate(size_groups):
            mean_area = group_area / count
            
            # If area is within range of an existing group, increment the count
            if abs(mean_area - area) <= 500:
                size_groups[idx] = (group_area + area, count + 1)
                found_group = True
                break

        if not found_group:
            size_groups.append((area, 1))

    # Draw contours without adding any text on the object
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area < min_area_threshold:
            continue

        # Determine which size group this contour belongs to
        for idx, (group_area, count) in enumerate(size_groups):
            mean_area = group_area / count
            if abs(mean_area - area) <= 800:
                color = stable_colors[idx % len(stable_colors)]
                cv2.drawContours(frame, [contour], -1, color, 2)
                break

    # Display size group counts and mean areas in the upper-left corner
    y0, dy = 50, 30
    for i, (group_area, count) in enumerate(size_groups):
        mean_area = group_area / count
        size_label = f"Size {chr(65 + i)}: {count} (Mean Area: {mean_area:.1f})"
        color = stable_colors[i % len(stable_colors)]
        cv2.putText(frame, size_label, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return frame

# Streamlit app setup
st.title("Real-Time Object Detection and Grouping")

# Start capturing video stream from the webcam
run = st.checkbox('Run')
camera = cv2.VideoCapture(0)

# Stream video frames in real-time
frame_window = st.image([])

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Failed to capture image")
        break

    # Process the frame
    processed_frame = process_liveDynamic(frame)

    # Convert the frame from BGR to RGB for display in Streamlit
    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

    # Display the frame in the Streamlit app
    frame_window.image(frame_rgb)

# Release the camera when done
camera.release()
