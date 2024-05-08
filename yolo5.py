import torch
import cv2
import numpy as np

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('ROI')
cv2.setMouseCallback('ROI', POINTS)

cap=cv2.VideoCapture('dataset/vid1.mp4')

model = torch.hub.load("yolov5", "yolov5s", source="local")
count=0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    
    frame=cv2.resize(frame,(1020,500))

    roi = frame[64:273,297:690]
    results = model(roi)

    x1_roi, y1_roi = 297, 64  # Top-left corner
    x2_roi, y2_roi = 690, 273  # Bottom-right corner
    corners = [(x1_roi, y1_roi), (x1_roi, y2_roi), (x2_roi, y2_roi), (x2_roi, y1_roi)]

    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d=(row['name'])
        # print(d)
        cv2.rectangle(roi,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.polylines(frame, [np.array(corners)], True, (0, 255, 0), 3)
    cv2.imshow("ROI",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
