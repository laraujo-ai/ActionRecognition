import numpy as np
import cv2
from typing import Tuple


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


#pose estimator functions
def letterbox_simple(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (480, 480),
    color: Tuple[int, int, int] = (114, 114, 114),
    scale_up: bool = True,
) -> Tuple[np.ndarray, float, Tuple[int, int, int, int]]:
    """
    Simple letterbox: resize keeping aspect ratio, then pad (left,top,right,bottom).
    Returns: (img_letterboxed, scale, (left,top,right,bottom))
    """
    h, w = img.shape[:2]
    nw, nh = new_shape[1], new_shape[0]
    scale = min(nw / w, nh / h)
    if not scale_up:
        scale = min(scale, 1.0)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    if (new_w, new_h) != (w, h):
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img.copy()

    # compute padding
    dw = nw - new_w
    dh = nh - new_h
    left = dw // 2
    right = dw - left
    top = dh // 2
    bottom = dh - top

    img_padded = cv2.copyMakeBorder(
        img_resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

    img_padded = (
        cv2.resize(img_padded, (nw, nh))
        if img_padded.shape[:2] != (nh, nw)
        else img_padded
    )

    return img_padded, float(scale), (left, top, right, bottom)


def nms(boxes, scores, threshold=0.5):
    # Sort the bounding boxes by their confidence scores in descending order
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    
    # Initialize a list of selected bounding box indices
    selected_indices = []
    
    # Loop over the sorted list of bounding boxes
    while len(boxes) > 0:
        # Select the box or keypoint with the highest confidence score
        selected_index = indices[0]
        selected_indices.append(selected_index)
        
        # Compute the overlap between the selected box and all other boxes
        ious = compute_iou(boxes[0], boxes[1:])
        
        # Remove all boxes that overlap with the selected box by more than the threshold
        indices = indices[1:][ious <= threshold]
        boxes = boxes[1:][ious <= threshold]
    
    # Convert the list of selected box indices to an array and return it
    return np.array(selected_indices)

def compute_iou(box, boxes):
    """Compute the intersection-over-union (IoU) between a bounding box and a set of other bounding boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - intersection
    return intersection / union

def non_max_suppression(prediction, conf_thres=0.05):
    confidences = prediction[4, :]
    conf_mask = confidences > conf_thres
    x = (prediction.T)[conf_mask]
    if x.shape[0] == 0:
        return [], [], []

    box, conf, mask = np.split(x, (4, 5), axis=1)
    boxxyxy = np.copy(box)
    boxxyxy[..., 0] = box[..., 0] - box[..., 2] / 2  # top left x
    boxxyxy[..., 1] = box[..., 1] - box[..., 3] / 2  # top left y
    boxxyxy[..., 2] = box[..., 0] + box[..., 2] / 2  # bottom right x
    boxxyxy[..., 3] = box[..., 1] + box[..., 3] / 2  # bottom right y
    box = boxxyxy

    x = np.concatenate((box, conf, mask), axis=1)

    x = x[x[:, 4].argsort()[::-1]]
    boxes, scores = x[:, :4], x[:, 4]

    indices = nms(boxes, scores)
    x = x[indices]

    return x[:, :4], x[:, 4], x[:, 5:]