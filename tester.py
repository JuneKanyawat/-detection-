import cv2
import numpy as np
import pickle
from skimage.transform import resize
import tkinter as tk

# Define constants for spot status
EMPTY = False
NOT_EMPTY = True

# Load pre-trained model
model = pickle.load(open("datasets/model/partsix_model.p", "rb"))

# Function to determine if a spot is empty or not using the pre-trained model
def empty_or_not(spot_bgr):
    flat_data = []
    img_resized = resize(spot_bgr, (30, 10, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_output = model.predict(flat_data)
    return EMPTY if y_output == 0 else NOT_EMPTY

# Function to extract bounding boxes for spots from connected components
def get_spots_boxes(connected_components):
    totalLabels, label_ids, values, centroid = connected_components
    slots = []
    for i in range(1, totalLabels):
        x1 = int(values[i, cv2.CC_STAT_LEFT])
        y1 = int(values[i, cv2.CC_STAT_TOP])
        w = int(values[i, cv2.CC_STAT_WIDTH])
        h = int(values[i, cv2.CC_STAT_HEIGHT])
        slots.append([x1, y1, w, h])
    return slots

# Function to calculate the difference between two images
def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# Main function to process video frames and handle mask movement
def process_frames(mask_pos, root):
    # Paths to mask image and video
    mask_path = 'datasets/frame/working_space.png'
    video_path = 'datasets/video/video_p1.mp4'

    # Load mask image in grayscale
    mask = cv2.imread(mask_path, 0)

    # Capture video
    cap = cv2.VideoCapture(video_path)

    # Get connected components from the mask image and extract spot bounding boxes
    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    spots = get_spots_boxes(connected_components)

    # Initialize variables to track spot status and differences
    spots_status = [None for _ in spots]
    diffs = [None for _ in spots]
    previous_frame = None

    frame_nmr = 0
    ret = True
    step = 30
    paused = False

    # Function to apply mask offset to spots
    def apply_mask_offset(spots, offset):
        return [[x + offset[1], y + offset[0], w, h] for x, y, w, h in spots]

    # Function to handle button clicks
    def move_mask(direction):
        nonlocal mask_pos
        if direction == 'up':
            mask_pos[0] -= 1
        elif direction == 'down':
            mask_pos[0] += 1
        elif direction == 'left':
            mask_pos[1] -= 1
        elif direction == 'right':
            mask_pos[1] += 1

    # Create buttons for movement
    def create_button(text, direction):
        button = tk.Button(root, text=text, command=lambda: move_mask(direction))
        button.pack()

    # Function to toggle between play and pause
    def toggle_pause():
        nonlocal paused
        paused = not paused

    # Create buttons for movement
    create_button("Up", 'up')
    create_button("Down", 'down')
    create_button("Left", 'left')
    create_button("Right", 'right')
    play_button = tk.Button(root, text="Play/Pause", command=toggle_pause)
    play_button.pack()

    # Main loop to process video frames
    while ret:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply the mask offset
            adjusted_spots = apply_mask_offset(spots, mask_pos)

            if frame_nmr % step == 0 and previous_frame is not None:
                for spot_indx, spot in enumerate(adjusted_spots):
                    x1, y1, w, h = spot
                    spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                    diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])
                print([diffs[j] for j in np.argsort(diffs)][::-1])

            if frame_nmr % step == 0:
                arr_ = range(len(adjusted_spots)) if previous_frame is None else [j for j in np.argsort(diffs) if
                                                                                  diffs[j] / np.amax(diffs) > 0.4]
                for spot_indx in arr_:
                    spot = adjusted_spots[spot_indx]
                    x1, y1, w, h = spot
                    spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                    spot_status = empty_or_not(spot_crop)
                    spots_status[spot_indx] = spot_status

            if frame_nmr % step == 0:
                previous_frame = frame.copy()

            for spot_indx, spot in enumerate(adjusted_spots):
                spot_status = spots_status[spot_indx]
                x1, y1, w, h = adjusted_spots[spot_indx]
                color = (0, 255, 0) if spot_status else (0, 0, 255)
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

            cv2.rectangle(frame, (80, 20), (420, 80), (255, 255, 255), -1)
            cv2.putText(frame, 'Available parts: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))),
                        (100, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)

            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.imshow('frame', frame)

            frame_nmr += 1

        # Update tkinter window
        root.update()

        # Check for quit event
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create tkinter window
root = tk.Tk()
root.title("Mask Movement")

# Initial mask position
mask_pos = [0, 0]

# Call the main function to start processing frames
process_frames(mask_pos, root)

# Start tkinter main loop
root.mainloop()
