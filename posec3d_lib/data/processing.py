import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Union, List, Sequence, Any, Mapping


def pseudo_collate(data_batch: Sequence) -> Any:
    """Convert list of data sampled from dataset into a batch of data, of which
    type consistent with the type of each data_item in ``data_batch``.

    Args:
        data_batch (Sequence): Batch of data from dataloader.

    Returns:
        Any: Transversed Data in the same format as the data_itement of
        ``data_batch``.
    """  # noqa: E501
    data_item = data_batch[0]
    data_item_type = type(data_item)

    if isinstance(data_item, Sequence):
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError(
                "each data_itement in list of batch should be of equal size"
            )
        transposed = list(zip(*data_batch))
        if isinstance(data_item, tuple):
            return [pseudo_collate(samples) for samples in transposed]
        else:
            try:
                return data_item_type(
                    [pseudo_collate(samples) for samples in transposed]
                )
            except TypeError:
                return [pseudo_collate(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        return data_item_type(
            {key: pseudo_collate([d[key] for d in data_batch]) for key in data_item}
        )
    else:
        return data_batch


def rescale_size(
    old_size: Tuple[int, int],
    scale: Union[float, int, Tuple[int, int]],
    return_scale: bool = False,
) -> Union[Tuple[int, int], Tuple[Tuple[int, int], float]]:
    """
    Calculate the new size to be rescaled to.

    Args:
        old_size (Tuple[int, int]): The old size (w, h) of image.
        scale (float | int | Tuple[int, int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size. Defaults to False.

    Returns:
        Tuple[int, int] | Tuple[Tuple[int, int], float]: The new rescaled image size,
        and optionally the scaling factor.

    Examples:
        >>> rescale_size((640, 480), 0.5)
        (320, 240)
        >>> rescale_size((640, 480), (320, 240))
        (320, 240)
        >>> rescale_size((640, 480), (-1, 64))
        (64, 48)
    """
    w, h = old_size

    if isinstance(scale, tuple):
        if len(scale) != 2:
            raise ValueError(f"Scale tuple must have 2 elements, got {len(scale)}")

        max_long_edge = max(scale)
        max_short_edge = min(scale)
        if max_short_edge == -1:
            if w >= h:
                # Width is longer edge
                actual_scale = max_long_edge / w
            else:
                # Height is longer edge
                actual_scale = max_long_edge / h
            new_w = int(w * actual_scale + 0.5)
            new_h = int(h * actual_scale + 0.5)
        else:
            scale_w = scale[0] / w
            scale_h = scale[1] / h
            actual_scale = min(scale_w, scale_h)
            new_w = int(w * actual_scale + 0.5)
            new_h = int(h * actual_scale + 0.5)
    else:
        raise TypeError(
            f"Scale must be float, int or tuple of int, but got {type(scale)}"
        )

    new_size = (new_w, new_h)
    return new_size


def resize_images(
    images: List[np.ndarray], new_size: Tuple[int, int], interpolation: str = "bilinear"
) -> List[np.ndarray]:
    """
    Resize a list of images to the specified size.

    Args:
        images (List[np.ndarray]): List of images to resize
        new_size (Tuple[int, int]): Target size (width, height)
        interpolation (str): Interpolation method. Options:
            "nearest", "bilinear", "bicubic", "area", "lanczos". Default: "bilinear"

    Returns:
        List[np.ndarray]: List of resized images
    """
    # Map interpolation string to OpenCV constants
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    if interpolation not in interpolation_map:
        raise ValueError(
            f"Unsupported interpolation: {interpolation}. "
            f"Supported: {list(interpolation_map.keys())}"
        )

    cv_interpolation = interpolation_map[interpolation]
    new_w, new_h = new_size

    resized_images = []
    for img in images:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv_interpolation)
        resized_images.append(resized)

    return resized_images


def resize_keypoints(keypoints: np.ndarray, scale_factor: np.ndarray) -> np.ndarray:
    """
    Resize keypoints according to scale factor.

    Args:
        keypoints (np.ndarray): Keypoints with shape (..., 2) or (..., 3)
        scale_factor (np.ndarray): Scale factors [scale_x, scale_y]

    Returns:
        np.ndarray: Resized keypoints
    """
    resized_kps = keypoints.copy()
    resized_kps[..., 0] *= scale_factor[0]  # x coordinates
    resized_kps[..., 1] *= scale_factor[1]  # y coordinates
    return resized_kps


def resize_bboxes(bboxes: np.ndarray, scale_factor: np.ndarray) -> np.ndarray:
    """
    Resize bounding boxes according to scale factor.

    Args:
        bboxes (np.ndarray): Bounding boxes with shape (N, 4) in format [x1, y1, x2, y2]
        scale_factor (np.ndarray): Scale factors [scale_x, scale_y]

    Returns:
        np.ndarray: Resized bounding boxes
    """
    if len(scale_factor) != 2:
        raise ValueError(f"scale_factor must have length 2, got {len(scale_factor)}")

    # Extend scale_factor to [scale_x, scale_y, scale_x, scale_y] for [x1, y1, x2, y2]
    full_scale_factor = np.concatenate([scale_factor, scale_factor])
    return bboxes * full_scale_factor


def resize_from_data(
    data: Dict,
    scale: Union[float, int, Tuple[int, int]],
    interpolation: str = "bilinear",
) -> Dict:
    """
    Apply resizing to data dictionary.

    Args:
        data (Dict): Data dictionary containing:
            - img_shape (Tuple[int, int]): Image shape (height, width)
            - keypoint (np.ndarray, optional): Keypoint data to resize
            - imgs (List[np.ndarray], optional): Images to resize
            - scale_factor (np.ndarray, optional): Previous scale factors
        scale (float | int | Tuple[int, int]): Scale parameter
        interpolation (str): Interpolation method. Default: "bilinear".

    Returns:
        Dict: Updated data dictionary with modified keys:
            - img_shape: New image shape after resizing
            - keypoint: Resized keypoint coordinates (if present)
            - imgs: Resized images (if present)
            - scale_factor: Cumulative scale factors

    Example:
        >>> data = {
        ...     "img_shape": (480, 640),
        ...     "keypoint": np.random.rand(1, 50, 17, 2) * 100
        ... }
        >>> result = resize_from_data(data, scale=(-1, 64), keep_ratio=True)
        >>> print(result["img_shape"])  # New dimensions
    """
    img_shape = data["img_shape"]

    if "scale_factor" not in data:
        data["scale_factor"] = np.array([1.0, 1.0], dtype=np.float32)

    img_h, img_w = img_shape
    new_w, new_h = rescale_size((img_w, img_h), scale)
    current_scale_factor = np.array([new_w / img_w, new_h / img_h], dtype=np.float32)

    data["img_shape"] = (new_h, new_w)
    data["scale_factor"] = data["scale_factor"] * current_scale_factor

    if "keypoint" in data:
        data["keypoint"] = resize_keypoints(data["keypoint"], current_scale_factor)

    if "imgs" in data:
        data["imgs"] = resize_images(data["imgs"], (new_w, new_h), interpolation)

    return data


def _combine_quadruple(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Combine two crop quadruples.

    This replicates MMAction2's _combine_quadruple function behavior.
    Each quadruple represents [x, y, w, h] as normalized coordinates.

    Args:
        a (np.ndarray): First quadruple [x, y, w, h]
        b (np.ndarray): Second quadruple [x, y, w, h]

    Returns:
        np.ndarray: Combined quadruple
    """
    a_x, a_y, a_w, a_h = a
    b_x, b_y, b_w, b_h = b

    # Combine the transformations
    new_x = a_x + b_x * a_w
    new_y = a_y + b_y * a_h
    new_w = a_w * b_w
    new_h = a_h * b_h

    return np.array([new_x, new_y, new_w, new_h], dtype=np.float32)


def crop_keypoints(keypoints: np.ndarray, crop_bbox: np.ndarray) -> np.ndarray:
    """
    Crop keypoints according to crop_bbox.

    Args:
        keypoints (np.ndarray): Keypoints to crop
        crop_bbox (np.ndarray): Crop bounding box [left, top, right, bottom]

    Returns:
        np.ndarray: Cropped keypoints
    """
    # Subtract the top-left corner of crop region
    cropped_kps = keypoints.copy()
    cropped_kps[..., 0] -= crop_bbox[0]  # x coordinates
    cropped_kps[..., 1] -= crop_bbox[1]  # y coordinates
    return cropped_kps


def crop_images(images: List[np.ndarray], crop_bbox: np.ndarray) -> List[np.ndarray]:
    """
    Crop images according to crop_bbox.

    Args:
        images (List[np.ndarray]): Images to crop
        crop_bbox (np.ndarray): Crop bounding box [left, top, right, bottom]

    Returns:
        List[np.ndarray]: Cropped images
    """
    left, top, right, bottom = crop_bbox.astype(int)
    return [img[top:bottom, left:right] for img in images]


def crop_bboxes(bboxes: np.ndarray, crop_bbox: np.ndarray) -> np.ndarray:
    """
    Crop bounding boxes according to crop_bbox.

    Args:
        bboxes (np.ndarray): Bounding boxes to crop with shape (N, 4)
        crop_bbox (np.ndarray): Crop bounding box [left, top, right, bottom]

    Returns:
        np.ndarray: Cropped bounding boxes
    """
    left, top, right, bottom = crop_bbox
    crop_w, crop_h = right - left, bottom - top

    # Adjust bounding boxes relative to crop region
    cropped_bboxes = bboxes.copy()
    cropped_bboxes[..., 0::2] = np.clip(
        bboxes[..., 0::2] - left, 0, crop_w - 1
    )  # x coordinates
    cropped_bboxes[..., 1::2] = np.clip(
        bboxes[..., 1::2] - top, 0, crop_h - 1
    )  # y coordinates

    return cropped_bboxes


def center_crop_from_data(data: Dict, crop_size: Union[int, Tuple[int, int]]) -> Dict:
    """
    Apply center cropping to data dictionary.

    Args:
        data (Dict): Data dictionary containing:
            - img_shape (Tuple[int, int]): Image shape (height, width)
            - keypoint (np.ndarray, optional): Keypoint data to crop
            - imgs (List[np.ndarray], optional): Images to crop
            - crop_quadruple (np.ndarray, optional): Previous crop quadruple
        crop_size (int | Tuple[int, int]): (w, h) of crop size

    Returns:
        Dict: Updated data dictionary with modified keys:
            - img_shape: New image shape after cropping
            - keypoint: Cropped keypoint coordinates (if present)
            - imgs: Cropped images (if present)
            - crop_bbox: Crop bounding box applied
            - crop_quadruple: Updated crop quadruple

    Example:
        >>> data = {
        ...     "img_shape": (480, 640),
        ...     "keypoint": np.random.rand(1, 50, 17, 2) * np.array([640, 480])
        ... }
        >>> result = center_crop_from_data(data, crop_size=224)
        >>> print(result["img_shape"])  # Should be (224, 224)
    """
    img_shape = data["img_shape"]
    img_h, img_w = img_shape

    # Handle crop_size input
    if isinstance(crop_size, int):
        crop_w, crop_h = crop_size, crop_size
    elif isinstance(crop_size, (tuple, list)) and len(crop_size) == 2:
        crop_w, crop_h = crop_size
    else:
        raise TypeError(
            f"crop_size must be int or tuple of int, but got {type(crop_size)}"
        )

    # Calculate center crop coordinates
    left = (img_w - crop_w) // 2
    top = (img_h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h
    new_h, new_w = bottom - top, right - left

    # Create crop bbox
    crop_bbox = np.array([left, top, right, bottom], dtype=np.float32)
    data["crop_bbox"] = crop_bbox
    data["img_shape"] = (new_h, new_w)

    # Initialize crop_quadruple if not present
    if "crop_quadruple" not in data:
        data["crop_quadruple"] = np.array([0, 0, 1, 1], dtype=np.float32)  # x, y, w, h

    # Calculate new crop quadruple
    x_ratio = left / img_w
    y_ratio = top / img_h
    w_ratio = new_w / img_w
    h_ratio = new_h / img_h

    old_crop_quadruple = data["crop_quadruple"]
    old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
    old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]

    # Combine crop quadruples
    new_crop_quadruple = np.array(
        [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio,
            w_ratio * old_w_ratio,
            h_ratio * old_h_ratio,
        ],
        dtype=np.float32,
    )
    data["crop_quadruple"] = new_crop_quadruple

    # Apply cropping to data elements
    if "keypoint" in data:
        data["keypoint"] = crop_keypoints(data["keypoint"], crop_bbox)

    if "imgs" in data:
        data["imgs"] = crop_images(data["imgs"], crop_bbox)

    return data
