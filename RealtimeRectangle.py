import cv2
import streamlit as st

def live_rectangle_detection():
    # Initialize video capture for webcam
    cap = cv2.VideoCapture(0)  # 0 for the default webcam

    if not cap.isOpened():
        st.error("Error: Unable to access webcam.")
        return

    min_area = 1000  # Set minimum area for rectangle detection
    
    # Streamlit frame
    st.title("Live Rectangle Detection")
    stframe = st.empty()

    # Loop to continuously get frames from the webcam and process them
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam")
            break

        # Convert to grayscale and threshold the frame for contour detection
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(frame_gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over all contours and find rectangles
        rectangle_count = 0
        output_frame = frame.copy()

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4 and cv2.contourArea(contour) > min_area:
                cv2.drawContours(output_frame, [approx], 0, (0, 255, 0), 2)
                rectangle_count += 1

        # Display the total count of rectangles on the frame
        cv2.putText(output_frame, f"Total Rectangles: {rectangle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert the output frame to RGB (since OpenCV uses BGR)
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(output_frame, channels="RGB")

    # Release the video capture object when the loop is done (if it exits)
    cap.release()

if __name__ == '__main__':
    live_rectangle_detection()