import cv2
import numpy as np
import pickle
from skimage.transform import resize
import time
from datetime import datetime

# Load models
model1 = pickle.load(open("partsix_model.p", "rb"))
model2 = pickle.load(open("partone_model.p", "rb"))
model3 = pickle.load(open("parttwo_model.p", "rb"))
model4 = pickle.load(open("partthree_model.p", "rb"))  # Added loading the fourth model
model5 = pickle.load(open("handtwo_model.p", "rb"))

# Define constants for categories
EMPTY = 0
NOT_EMPTY = 1
OTHER = 2  # Add more categories as needed

def get_spots_boxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
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
video_path = 'datasets/video/video_p1.mp4'

# Load mask images
mask1_img = cv2.imread(mask1, 0)
mask2_img = cv2.imread(mask2, 0)
mask3_img = cv2.imread(mask3, 0)
mask4_img = cv2.imread(mask4, 0)
mask5_img = cv2.imread(mask5, 0)

# Get spots from masks
connected_components_mask1 = cv2.connectedComponentsWithStats(mask1_img, 4, cv2.CV_32S)
spots1 = get_spots_boxes(connected_components_mask1)
connected_components_mask2 = cv2.connectedComponentsWithStats(mask2_img, 4, cv2.CV_32S)
spots2 = get_spots_boxes(connected_components_mask2)
connected_components_mask3 = cv2.connectedComponentsWithStats(mask3_img, 4, cv2.CV_32S)
spots3 = get_spots_boxes(connected_components_mask3)
connected_components_mask4 = cv2.connectedComponentsWithStats(mask4_img, 4, cv2.CV_32S)  # Processed the new mask
spots4 = get_spots_boxes(connected_components_mask4)
connected_components_mask5 = cv2.connectedComponentsWithStats(mask5_img, 4, cv2.CV_32S)  # Processed the fifth mask
spots5 = get_spots_boxes(connected_components_mask5)

# Initialize status and diffs arrays
spots_status1 = [None for _ in spots1]
spots_status2 = [None for _ in spots2]
spots_status3 = [None for _ in spots3]
spots_status4 = [None for _ in spots4]
spots_status5 = [None for _ in spots5]
diffs1 = [None for _ in spots1]
diffs2 = [None for _ in spots2]
diffs3 = [None for _ in spots3]
diffs4 = [None for _ in spots4]
diffs5 = [None for _ in spots5]

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

# Variables to store cycle time and assembly time
cycle_time = "N/A"
assembly_time = "N/A"

# Open video capture
cap = cv2.VideoCapture(video_path)
frame_nmr = 0
ret = True
step = 10
cycle_counter = 1  # Initialize cycle counter
x2 = 0
y2 = 0

# Initialize the plastic flag
plastic = False
newbook = False
dryer_part1 = False
dryer_part2 = False

while ret:
    ret, frame = cap.read()

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots1):
            x2,y2,w,h = spot
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs1[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        for spot_indx, spot in enumerate(spots2):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs2[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        for spot_indx, spot in enumerate(spots3):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs3[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        for spot_indx, spot in enumerate(spots4):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs4[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        for spot_indx, spot in enumerate(spots5):  # Calculated diffs for the fifth spots
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs5[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_1 = range(len(spots1))
            arr_2 = range(len(spots2))
            arr_3 = range(len(spots3))
            arr_4 = range(len(spots4))
            arr_5 = range(len(spots5))
        else:
            arr_1 = [j for j in np.argsort(diffs1) if diffs1[j] / np.amax(diffs1) > 0.4]
            arr_2 = [j for j in np.argsort(diffs2) if diffs2[j] / np.amax(diffs2) > 0.4]
            arr_3 = [j for j in np.argsort(diffs3) if diffs3[j] / np.amax(diffs3) > 0.4]
            arr_4 = [j for j in np.argsort(diffs4) if diffs4[j] / np.amax(diffs4) > 0.4]
            arr_5 = [j for j in np.argsort(diffs5) if diffs5[j] / np.amax(diffs5) > 0.4]

        for spot_indx in arr_1:
            spot = spots1[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop, model1)
            spots_status1[spot_indx] = spot_status

        for spot_indx in arr_2:
            spot = spots2[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop, model2)
            spots_status2[spot_indx] = spot_status

        for spot_indx in arr_3:
            spot = spots3[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop, model3)
            spots_status3[spot_indx] = spot_status

        for spot_indx in arr_4:  # Processed the new spots
            spot = spots4[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop, model4)
            spots_status4[spot_indx] = spot_status

        for spot_indx in arr_5:  # Processed each spot for the fifth mask
            spot = spots5[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop, model5)
            spots_status5[spot_indx] = spot_status

    # Detecting stages E and S, and print time when decrement in mask 3 occurs
    if frame_nmr % step == 0:
        not_empty_count2 = sum(1 for s in spots_status2 if s == NOT_EMPTY)
        not_empty_count1 = sum(1 for s in spots_status1 if s == NOT_EMPTY)
        not_empty_count3 = sum(1 for s in spots_status3 if s == NOT_EMPTY)
        not_empty_count4 = sum(1 for s in spots_status4 if s == NOT_EMPTY)
        not_empty_count5 = sum(1 for s in spots_status5 if s == NOT_EMPTY)# Added count for the new spots

        # Detect S stage (decrement in mask 1)
        if not s_detected and not_empty_count1 < sum(1 for s in previous_spots_status1 if s == NOT_EMPTY):
            t_S = time.time() - start_time
            s_detected = True
            e_detected = False  # Reset E detection flag for a new cycle
            print("S stage detected at: {:.1f} seconds".format(t_S))

        # Detect E stage (increment in mask 2)
        if not e_detected and not_empty_count2 > sum(1 for s in previous_spots_status2 if s == NOT_EMPTY):
            t_E = time.time() - start_time
            e_detected = True
            print("E stage detected at: {:.1f} seconds".format(t_E))

            # Reset the plastic flag to False
            plastic = False
            newbook = False

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
        if s_detected and not plastic and not_empty_count3 < sum(1 for s in previous_spots_status3 if s == NOT_EMPTY):
            # Print the time when decrement in mask 3 occurs
            plastic = True
            print("Decrement in mask 3 detected at: {:.1f} seconds".format(time.time() - start_time))

        # Detect decrement in mask 4
        if s_detected and not newbook and not_empty_count4 < sum(1 for s in previous_spots_status4 if s == NOT_EMPTY):
            # Print the time when decrement in mask 4 occurs
            newbook = True
            print("Decrement in mask 4 detected at: {:.1f} seconds".format(time.time() - start_time))

        not_empty_count5 = sum(1 for s in spots_status5 if s == NOT_EMPTY)
        previous_not_empty_count5 = sum(1 for s in previous_spots_status5 if s == NOT_EMPTY)

        if s_detected and not dryer_part1 and not_empty_count5 < previous_not_empty_count5:
            if spots_status5[1] == EMPTY and previous_spots_status5[1] == NOT_EMPTY:
                dryer_part1 = True
                print("Dryer part 1 became empty at: {:.1f} seconds".format(time.time() - start_time))
            if all(s == EMPTY for s in spots_status5) and any(s == NOT_EMPTY for s in previous_spots_status5):
                dryer_part2 = True
                print("Dryer part 2 became empty at: {:.1f} seconds".format(time.time() - start_time))

        # Update previous statuses
    if frame_nmr % step == 0:
        previous_frame = frame.copy()
        previous_spots_status1 = spots_status1.copy()
        previous_spots_status2 = spots_status2.copy()
        previous_spots_status3 = spots_status3.copy()
        previous_spots_status4 = spots_status4.copy()
        previous_spots_status5 = spots_status5.copy()# Updated the previous status array for the new spots

        # Drawing rectangles and texts
    for spot_indx, spot in enumerate(spots1):
        spot_status = spots_status1[spot_indx]
        x1, y1, w, h = spots1[spot_indx]
        if spot_status == EMPTY:
            color = (0, 0, 255)  # Red
        elif spot_status == NOT_EMPTY:
            color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Blue or another color for additional categories
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    for spot_indx, spot in enumerate(spots2):
        spot_status = spots_status2[spot_indx]
        x1, y1, w, h = spots2[spot_indx]
        if spot_status == EMPTY:
            color = (0, 0, 255)  # Red
        elif spot_status == NOT_EMPTY:
            color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Blue or another color for additional categories
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    for spot_indx, spot in enumerate(spots3):
        spot_status = spots_status3[spot_indx]
        x1, y1, w, h = spots3[spot_indx]
        if spot_status == EMPTY:
            color = (0, 0, 255)  # Red
        elif spot_status == NOT_EMPTY:
            color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Blue or another color for additional categories
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    for spot_indx, spot in enumerate(spots4):  # Processed the new spots
        spot_status = spots_status4[spot_indx]
        x1, y1, w, h = spots4[spot_indx]
        if spot_status == EMPTY:
            color = (0, 0, 255)  # Red
        elif spot_status == NOT_EMPTY:
            color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Blue or another color for additional categories
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    for spot_indx, spot in enumerate(spots5):  # Processed the new spots
        spot_status = spots_status5[spot_indx]
        x1, y1, w, h = spots5[spot_indx]
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

    # Add the current time text to the frame
    cv2.putText(frame, current_time, (frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

    # Display the available parts info
    cv2.rectangle(frame, (40, 0), (300, 80), (255, 255, 255), -1)

    # Display the cycle time and assembly time above the available parts info
    cv2.putText(frame, 'Cycle time: {}'.format(cycle_time), (60, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(frame, 'Assembly time: {}'.format(assembly_time), (60, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)

    cv2.putText(frame,
                '{} / {}'.format(str(sum([1 for s in spots_status1 if s == NOT_EMPTY])), str(len(spots_status1))),
                (x2, y2 + 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)

    cv2.putText(frame,
                '{} / {}'.format(str(sum([1 for s in spots_status2 if s == NOT_EMPTY])), str(len(spots_status2))),
                (x1 - 100, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)

    cv2.putText(frame,
                '{} / {}'.format(str(sum([1 for s in spots_status3 if s == NOT_EMPTY])), str(len(spots_status3))),
                (x1 - 100, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)

    cv2.putText(frame,
                '{} / {}'.format(str(sum([1 for s in spots_status4 if s == NOT_EMPTY])), str(len(spots_status4))),
                (x1 - 100, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)  # Added info for the new spots

    cv2.putText(frame,
                '{} / {}'.format(str(sum([1 for s in spots_status5 if s == NOT_EMPTY])), str(len(spots_status5))),
                (x1 - 100, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)  # Added info for the new spots

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()


