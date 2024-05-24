import cv2
import numpy as np
import pickle
from skimage.transform import resize
import time
from datetime import datetime

# Load custom models for classification
model_paths = [
    "seven_model.p",
    "parttwo_model.p",
    "handtwo_model.p",
    "partthree_model.p",
    "partsix_model.p",
    "partone_model.p"
]
models = [pickle.load(open(mp, "rb")) for mp in model_paths]

# Define constants for categories
EMPTY = 0
NOT_EMPTY = 1
OTHER = 2  # Add more categories as needed

def get_spots_boxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components
    slots = []
    coef = 1
    for i in range(1, totalLabels):
        # Extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)
        slots.append([x1, y1, w, h])
    return slots

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

def empty_or_not(spot_bgr, model):
    flat_data = []
    img_resized = resize(spot_bgr, (30, 10, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_output = model.predict(flat_data)
    return y_output[0]

# Update paths for new data
mask_paths = [
    'datasets/frame/input_tray.png',
    'datasets/frame/plastic_bag.png',
    'datasets/frame/dryer_part.png',
    'datasets/frame/newbook.png',
    'datasets/frame/working_space.png',
    'datasets/frame/weight_scales.png'
]

# Load mask images
mask_imgs = [cv2.imread(mask, 0) for mask in mask_paths]

# Get spots from masks
connected_components_masks = [cv2.connectedComponentsWithStats(mask_img, 4, cv2.CV_32S) for mask_img in mask_imgs]
spots = [get_spots_boxes(cc) for cc in connected_components_masks]

# Initialize status and diffs arrays
spots_status = [[None for _ in spot] for spot in spots]
diffs = [[None for _ in spot] for spot in spots]

# Initialize timestamps and flags for stages
start_time = time.time()
t_E = t_S = t_prev_E = None
e_detected = s_detected = False

previous_frame = None
previous_spots_status = [[None for _ in spot] for spot in spots]

# Variables to store cycle time and assembly time
cycle_time = "N/A"
assembly_time = "N/A"

# Open video capture
video_path = 'datasets/video/video_p1.mp4'
cap = cv2.VideoCapture(video_path)
frame_nmr = 0
ret = True
step = 10
cycle_counter = 1  # Initialize cycle counter

# Initialize flags and timestamps for parts
input_tray = plastic_bag = dryer_part1 = dryer_part2 = newbook = working_space = weight_scales = False
timestamps = {
    "input_tray": None,
    "plastic_bag": None,
    "dryer_part1": None,
    "dryer_part2": None,
    "newbook": None,
    "working_space": None,
    "weight_scales": None
}

while ret:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_nmr % step == 0 and previous_frame is not None:
        for mask_idx, spot_list in enumerate(spots):
            for spot_indx, spot in enumerate(spot_list):
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                diffs[mask_idx][spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

    if frame_nmr % step == 0:
        for mask_idx, spot_list in enumerate(spots):
            if previous_frame is None:
                arr = range(len(spot_list))
            else:
                arr = [j for j in np.argsort(diffs[mask_idx]) if diffs[mask_idx][j] / np.amax(diffs[mask_idx]) > 0.4]

            for spot_indx in arr:
                x1, y1, w, h = spot_list[spot_indx]
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                spot_status = empty_or_not(spot_crop, models[mask_idx])
                spots_status[mask_idx][spot_indx] = spot_status

    # Detecting stages based on new logic
    if frame_nmr % step == 0:
        for mask_idx, spot_list in enumerate(spots):
            for spot_indx, spot in enumerate(spot_list):
                spot_status = spots_status[mask_idx][spot_indx]

                if mask_idx == 4:  # working_space 8
                    if spot_status == EMPTY and not working_space:
                        t_S = time.time() - start_time
                        working_space = True
                        timestamps["working_space"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # print(f"S stage detected at: {t_S:.1f} seconds, timestamp: {timestamps['working_space']}")
                        s_detected = True
                        e_detected = False  # Reset E detection flag for a new cycle

                # Logic for detecting stages based on specific spots
                if mask_idx == 1 and not plastic_bag:  # plastic_bag 3
                    if spot_status == EMPTY and working_space:
                        plastic_bag = True
                        timestamps["plastic_bag"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # print(f"plastic_bag detected at timestamp: {timestamps['plastic_bag']}")

                if mask_idx == 2:  # dryer_part 10 & 11
                    left_spot_status = spots_status[mask_idx][0]
                    right_spot_status = spots_status[mask_idx][1]
                    if right_spot_status == EMPTY and working_space:
                        dryer_part1 = True
                        timestamps["dryer_part1"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # print(f"dryer_part1 detected at timestamp: {timestamps['dryer_part1']}")

                    if dryer_part1 and (left_spot_status == EMPTY or (left_spot_status == EMPTY and right_spot_status == EMPTY)):
                        dryer_part2 = True
                        timestamps["dryer_part2"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # print(f"dryer_part2 detected at timestamp: {timestamps['dryer_part2']}")

                if mask_idx == 3:  # newbook 4
                    if spot_status == EMPTY and not newbook and plastic_bag and dryer_part1 and dryer_part2:
                        newbook = True
                        timestamps["newbook"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # print(f"newbook detected at timestamp: {timestamps['newbook']}")

                if mask_idx == 5:  # weight_scales 7
                    if spot_status == NOT_EMPTY and not weight_scales:
                        t_E = time.time() - start_time
                        weight_scales = True
                        timestamps["weight_scales"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"E stage detected at: {t_E:.1f} seconds, timestamp: {timestamps['weight_scales']}")
                        e_detected = True

                        # Calculate and store time between S and E stages if S was detected
                        if s_detected:
                            assembly_time = "{:.1f} s".format(t_E - t_S)
                            print(f"Assembly time {cycle_counter}: {assembly_time}")
                            cycle_counter += 1
                            s_detected = False  # Reset S detection flag for a new cycle

                        # Calculate and store time between two E stages
                        if t_prev_E is not None:
                            cycle_time = "{:.1f} s".format(t_E - t_prev_E)
                            print(f"Cycle time: {cycle_time}")

                        t_prev_E = t_E  # Update the previous E stage timestamp

                        # Check which components are missing
                        missing_components = {
                            "input_tray": input_tray,
                            "plastic_bag": plastic_bag,
                            "dryer_part1": dryer_part1,
                            "dryer_part2": dryer_part2,
                            "newbook": newbook,
                            "working_space": working_space,
                            "weight_scales": weight_scales
                        }

                        for component, status in missing_components.items():
                            if not status:
                                print(f"{component} missing")

                        # Reset all flags for a new cycle
                        input_tray = plastic_bag = dryer_part1 = dryer_part2 = newbook = working_space = weight_scales = False
                        timestamps = {key: None for key in timestamps}  # Reset timestamps

    if frame_nmr % step == 0:
        previous_frame = frame.copy()
        previous_spots_status = [status.copy() for status in spots_status]

    for mask_idx, spot_list in enumerate(spots):
        for spot_indx, spot in enumerate(spot_list):
            spot_status = spots_status[mask_idx][spot_indx]
            x1, y1, w, h = spot
            if spot_status == EMPTY:
                color = (0, 0, 255)  # Red
            elif spot_status == NOT_EMPTY:
                color = (0, 255, 0)  # Green
            else:
                color = (255, 0, 0)  # Blue or another color for additional categories
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    frame_nmr += 1

    # Get the current time
    current_time = datetime.now().strftime("%H:%M:%S")

    # Display the cycle time and assembly time above the available parts info
    cv2.putText(frame, 'Cycle time: {}'.format(cycle_time), (60, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(frame, 'Assembly time: {}'.format(assembly_time), (60, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)

    # Resize the frame for display
    display_height = 720  # or any desired height
    display_width = int(frame.shape[1] * (display_height / frame.shape[0]))
    frame_resized = cv2.resize(frame, (display_width, display_height))

    # Display the resized frame
    cv2.imshow('frame', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


