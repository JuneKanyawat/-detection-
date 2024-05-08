import torch

# Model
model = torch.hub.load("yolov5", "yolov5s", source="local")

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()
results.show()