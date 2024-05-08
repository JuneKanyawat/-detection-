import torch
import numpy as np
from skimage.transform import resize

# Assuming you have a PyTorch model loaded from model.pt
MODEL = torch.hub.load('yolov5', 'custom', 'best (1).pt', source="local")
MODEL.eval()

EMPTY = "Empty"
NOT_EMPTY = "Not Empty"

def empty_or_not(spot_bgr):
    flat_data = []

    img_resized = resize(spot_bgr, (30, 10, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    # Reshape flat_data to match the expected input shape of the model
    flat_data = flat_data.reshape(1, 3, 30, 10)  # Assuming 1 batch, 3 channels, 30 height, 10 width

    # Convert flat_data to a PyTorch tensor
    flat_data_tensor = torch.tensor(flat_data, dtype=torch.float32)

    # Perform prediction
    with torch.no_grad():
        y_output = MODEL(flat_data_tensor)

    # Assuming your model outputs probabilities for each class, and you want to consider it empty if the probability of the first class is greater than the second class
    prob_empty = y_output[0][0].item()
    prob_not_empty = y_output[0][1].item()

    if prob_empty > prob_not_empty:
        return EMPTY
    else:
        return NOT_EMPTY

# Example usage:
# result = empty_or_not(spot_bgr)
# print(result)