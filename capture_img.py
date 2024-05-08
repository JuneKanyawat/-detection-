import os
import cv2

# Open the video file
video_capture = cv2.VideoCapture('dataset/long-process-2.MP4')

# Check if the video file was successfully opened
if not video_capture.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

# Create a directory to save frames if it doesn't exist
output_directory = 'image_frames_02'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Read frames from the video
frame_count = 0
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    # Check if the frame was successfully read
    if not ret:
        break

    # Display the frame
    cv2.imshow('Frame', frame)

    # Save every third frame
    frame_count += 1
    if frame_count % 3 == 0:
        cv2.imwrite(os.path.join(output_directory, 'frame_%d.jpg' % (frame_count // 3)), frame)

    # Wait for a key press and exit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
