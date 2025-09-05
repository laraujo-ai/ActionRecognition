import numpy as np
import cv2
from typing import Tuple


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


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
