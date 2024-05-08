import torch
import cv2
import numpy as np

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('ROI')
cv2.setMouseCallback('ROI', POINTS)

cap = cv2.VideoCapture('dataset/long.MP4')

model = torch.hub.load("yolov5", "yolov5s", source="local")
count = 0

# Define the area of interest (ROI) coordinates
area = [(494, 315), (332, 341), (364, 470), (519, 444)]
x1_roi, y1_roi = min(area, key=lambda x: x[0])[0], min(area, key=lambda x: x[1])[1]
x2_roi, y2_roi = max(area, key=lambda x: x[0])[0], max(area, key=lambda x: x[1])[1]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]  # Extract ROI using area coordinates
    results = model(frame)


    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d = (row['name'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.polylines(frame, [np.array(area,np.int32)], True, (0, 255, 0), 3)
    cv2.imshow("ROI", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
