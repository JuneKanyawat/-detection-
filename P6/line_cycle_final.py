import cv2
import numpy as np
import pickle
from skimage.transform import resize
import time
from datetime import datetime

# Load models
model1 = pickle.load(open("datasets/model/partsix_model.p", "rb"))
model2 = pickle.load(open("datasets/model/partone_model.p", "rb"))
model3 = pickle.load(open("datasets/model/parttwo_model.p", "rb"))
model4 = pickle.load(open("datasets/model/partthree_model.p", "rb"))
model5 = pickle.load(open("datasets/model/handtwo_model.p", "rb"))
model6 = pickle.load(open("datasets/model/seven_model.p", "rb"))

# Define constants for categories
EMPTY = 0
NOT_EMPTY = 1
OTHER = 2

def get_spots_boxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components
    slots = []
    coef = 1
    for i in range(1, totalLabels):
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
mask1 = 'datasets/frame/working_space.png'
mask2 = 'datasets/frame/weight_scales.png'
mask3 = 'datasets/frame/plastic_bag.png'
mask4 = 'datasets/frame/newbook.png'
mask5 = 'datasets/frame/dryer_part.png'
mask6 = 'datasets/frame/input_tray.png'
video_path = 'datasets/video/DY08_P7_Full Clip.mp4'

# Load mask images
mask1_img = cv2.imread(mask1, 0)
mask2_img = cv2.imread(mask2, 0)
mask3_img = cv2.imread(mask3, 0)
mask4_img = cv2.imread(mask4, 0)
mask5_img = cv2.imread(mask5, 0)
mask6_img = cv2.imread(mask6, 0)

cap = cv2.VideoCapture(video_path)
# Load mask images and resize them to match the dimensions of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

mask1_img = cv2.resize(mask1_img, (frame_width, frame_height))
mask2_img = cv2.resize(mask2_img, (frame_width, frame_height))
mask3_img = cv2.resize(mask3_img, (frame_width, frame_height))
mask4_img = cv2.resize(mask4_img, (frame_width, frame_height))
mask5_img = cv2.resize(mask5_img, (frame_width, frame_height))
mask6_img = cv2.resize(mask6_img, (frame_width, frame_height))

# Get spots from masks
connected_components_mask1 = cv2.connectedComponentsWithStats(mask1_img, 4, cv2.CV_32S)
spots1 = get_spots_boxes(connected_components_mask1)
connected_components_mask2 = cv2.connectedComponentsWithStats(mask2_img, 4, cv2.CV_32S)
spots2 = get_spots_boxes(connected_components_mask2)
connected_components_mask3 = cv2.connectedComponentsWithStats(mask3_img, 4, cv2.CV_32S)
spots3 = get_spots_boxes(connected_components_mask3)
connected_components_mask4 = cv2.connectedComponentsWithStats(mask4_img, 4, cv2.CV_32S)
spots4 = get_spots_boxes(connected_components_mask4)
connected_components_mask5 = cv2.connectedComponentsWithStats(mask5_img, 4, cv2.CV_32S)
spots5 = get_spots_boxes(connected_components_mask5)
connected_components_mask6 = cv2.connectedComponentsWithStats(mask6_img, 4, cv2.CV_32S)
spots6 = get_spots_boxes(connected_components_mask6)

# Initialize status and diffs arrays
spots_status1 = [None for _ in spots1]
spots_status2 = [None for _ in spots2]
spots_status3 = [None for _ in spots3]
spots_status4 = [None for _ in spots4]
spots_status5 = [None for _ in spots5]
spots_status6 = [None for _ in spots6]
diffs1 = [None for _ in spots1]
diffs2 = [None for _ in spots2]
diffs3 = [None for _ in spots3]
diffs4 = [None for _ in spots4]
diffs5 = [None for _ in spots5]
diffs6 = [None for _ in spots6]

# Initialize timestamps and flags for stages
start_time = time.time()
t_E = t_S = t_prev_E = None
e_detected = s_detected = False

previous_frame = None
previous_spots_status1 = [None for _ in spots1]
previous_spots_status2 = [None for _ in spots2]
previous_spots_status3 = [None for _ in spots3]
previous_spots_status4 = [None for _ in spots4]
previous_spots_status5 = [None for _ in spots5]
previous_spots_status6 = [None for _ in spots6]

# Variables to store cycle time and assembly time
cycle_time = "N/A"
assembly_time = "N/A"

frame_nmr = 0
ret = True
step = 10
cycle_counter = 1  # Initialize cycle counter
x2 = 0
y2 = 0

# Initialize the flags
working_space = weight_scales = plastic = newbook = dryer_part1 = dryer_part2 = dryer = False
timestamps = {}

while ret:
    ret, frame = cap.read()

    if frame_nmr % step == 0 and previous_frame is not None:
        spot_diff_pairs = [(spots1, diffs1), (spots2, diffs2),
                           (spots3, diffs3), (spots4, diffs4),
                           (spots5, diffs5), (spots6, diffs6)]

        for spots, diffs in spot_diff_pairs:
            for spot_indx, spot in enumerate(spots):
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

    if frame_nmr % step == 0:
        arrs = []
        for diffs in [diffs1, diffs2, diffs3, diffs4, diffs5, diffs6]:
            if previous_frame is None:
                arr = range(len(diffs))
            else:
                arr = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
            arrs.append(arr)
        arr_1, arr_2, arr_3, arr_4, arr_5, arr_6 = arrs

        spot_model_pairs = [(spots1, spots_status1, model1), (spots2, spots_status2, model2),
                            (spots3, spots_status3, model3), (spots4, spots_status4, model4),
                            (spots5, spots_status5, model5), (spots6, spots_status6, model6)]

        for spots, spots_status, model in spot_model_pairs:
            for spot_indx in range(len(spots)):
                spot = spots[spot_indx]
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                spot_status = empty_or_not(spot_crop, model)
                spots_status[spot_indx] = spot_status

    # Detecting stages E and S, and print time when decrement in mask 3 occurs
    if frame_nmr % step == 0:
        not_empty_counts = []
        for spots_status in [spots_status1, spots_status2, spots_status3, spots_status4, spots_status5, spots_status6]:
            not_empty_counts.append(sum(1 for s in spots_status if s == NOT_EMPTY))

        not_empty_count1, not_empty_count2, not_empty_count3, not_empty_count4, not_empty_count5, not_empty_count6 = not_empty_counts

        # Detect S stage (decrement in mask 1)
        if not s_detected and not_empty_count1 < sum(1 for s in previous_spots_status1 if s == NOT_EMPTY):
            working_space = True
            timestamps["working_space"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            t_S = time.time() - start_time
            s_detected = True
            e_detected = False  # Reset E detection flag for a new cycle
            print("S stage detected at: {:.1f} seconds".format(t_S))

        # Detect E stage (increment in mask 2)
        if not e_detected and not_empty_count2 > sum(1 for s in previous_spots_status2 if s == NOT_EMPTY):
            weight_scales = True
            timestamps["weight_scales"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            t_E = time.time() - start_time
            e_detected = True
            print("E stage detected at: {:.1f} seconds".format(t_E))

            items_to_check = {
                "working_space": working_space, "dryer": dryer, "plastic": plastic, "dryer_part1": dryer_part1,
                "dryer_part2": dryer_part2, "newbook": newbook, "weight_scales": weight_scales
            }

            for item, present in items_to_check.items():
                if not present:
                    print(item, "missing")

            # Print the timestamps
            for key, value in timestamps.items():
                print("{} timestamp: {}".format(key, value))

            # Reset the plastic flag to False
            working_space = dryer = weight_scales = plastic = newbook = dryer_part1 = dryer_part2 = False
            timestamps = {}

            # Calculate and store time between S and E stages if S was detected
            if s_detected:
                assembly_time = "{:.1f} s".format(t_E - t_S)
                print("Assembly time {}: {}".format(cycle_counter, assembly_time))
                cycle_counter += 1
                s_detected = False  # Reset S detection flag for a new cycle

            # Calculate and store time between two E stages
            if t_prev_E is not None:
                cycle_time = "{:.1f} s".format(t_E - t_prev_E)
                print("Cycle time: {}".format(cycle_time))

            t_prev_E = t_E  # Update the previous E stage timestamp

        # Detect decrement in mask 3
        if s_detected and not plastic and working_space and not_empty_count3 < sum(1 for s in previous_spots_status3 if s == NOT_EMPTY):
            timestamps["plastic"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            plastic = True

        # Detect decrement in mask 4
        if s_detected and not newbook and plastic and dryer_part1 and dryer_part2 and not_empty_count4 < sum(1 for s in previous_spots_status4 if s == NOT_EMPTY):
            timestamps["newbook"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            newbook = True

        # Detect decrement in mask 6
        if s_detected and not dryer and working_space and not_empty_count6 < sum(1 for s in previous_spots_status6 if s == NOT_EMPTY):
            timestamps["dryer"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dryer = True

        not_empty_count5 = sum(1 for s in spots_status5 if s == NOT_EMPTY)
        previous_not_empty_count5 = sum(1 for s in previous_spots_status5 if s == NOT_EMPTY)

        if s_detected and not_empty_count5 < previous_not_empty_count5:
            if spots_status5[1] == EMPTY and previous_spots_status5[1] == NOT_EMPTY:
                timestamps["dryer_part1"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dryer_part1 = True
            if all(s == EMPTY for s in spots_status5) and any(s == NOT_EMPTY for s in previous_spots_status5):
                timestamps["dryer_part2"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dryer_part2 = True

    # Update previous statuses
    if frame_nmr % step == 0:
        previous_frame = frame.copy()
        previous_spots_status1 = spots_status1.copy()
        previous_spots_status2 = spots_status2.copy()
        previous_spots_status3 = spots_status3.copy()
        previous_spots_status4 = spots_status4.copy()
        previous_spots_status5 = spots_status5.copy()
        previous_spots_status6 = spots_status6.copy()

    # Drawing rectangles and texts
    spots_list = [(spots1, spots_status1), (spots2, spots_status2), (spots3, spots_status3), (spots4, spots_status4),
                  (spots5, spots_status5), (spots6, spots_status6)]

    for spots, spots_status in spots_list:
        for spot_indx, spot in enumerate(spots):
            spot_status = spots_status[spot_indx]
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
    cv2.putText(frame, current_time, (frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(frame, (40, 0), (300, 80), (255, 255, 255), -1)
    cv2.putText(frame, 'Cycle time: {}'.format(cycle_time), (60, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(frame, 'Assembly time: {}'.format(assembly_time), (60, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
    # Top group
    coords_texts_top = [((1, 80), (50, 120), '1', dryer), ((70, 80), (120, 120), '3', plastic),
                        ((140, 80), (190, 120), '4', newbook), ((210, 80), (260, 120), '7', weight_scales)]

    for (start, end, text, condition) in coords_texts_top:
        color = (0, 255, 0) if condition else (0, 0, 255)
        cv2.rectangle(frame, start, end, color, -1)
        text_pos = (start[0] + (end[0] - start[0]) // 2 - 10, start[1] + (end[1] - start[1]) // 2 + 10)
        cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    # Bottom group
    coords_texts_bottom = [((1, 150), (50, 190), '8', working_space), ((70, 150), (120, 190), '10', dryer_part1),
                           ((140, 150), (190, 190), '11', dryer_part2)]

    for (start, end, text, condition) in coords_texts_bottom:
        color = (0, 255, 0) if condition else (0, 0, 255)
        cv2.rectangle(frame, start, end, color, -1)
        text_pos = (start[0] + (end[0] - start[0]) // 2 - 10, start[1] + (end[1] - start[1]) // 2 + 10)
        cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()

