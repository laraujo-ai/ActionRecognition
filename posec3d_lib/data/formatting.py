import numpy as np
from typing import Dict, Union, Tuple, Optional, Sequence, Any


def _format_shape(
    imgs: np.ndarray,
    input_format: str,
    num_clips: int,
    clip_len: Union[int, Dict[str, int]],
    heatmap_imgs,
    modality,
) -> Tuple[
    np.ndarray, Tuple[int, ...], Optional[np.ndarray], Optional[Tuple[int, ...]]
]:
    """
    Format final imgs shape to the given input_format.

    Args:
        imgs (np.ndarray): Input images array
        input_format (str): Define the final data format. Options: "NCTHW", "NCHW", "NCTHW_Heatmap", "NPTCHW"
        num_clips (int): Number of clips
        clip_len (int | Dict[str, int]): Number of frames in each clip
        heatmap_imgs (np.ndarray, optional): Heatmap images array
        modality (str) : The modality in which the models run (pose, RBG, etc...)

    Returns:
        Tuple[np.ndarray, Tuple[int, ...], Optional[np.ndarray], Optional[Tuple[int, ...]]]:
            - imgs: Formatted images array
            - input_shape: Shape of formatted images
            - heatmap_imgs: Formatted heatmap images (if provided)
            - heatmap_input_shape: Shape of formatted heatmap images (if provided)

    """

    valid_formats = ["NCTHW", "NCTHW_Heatmap"]
    if input_format not in valid_formats:
        raise ValueError(
            f"The input format {input_format} is invalid. Valid formats: {valid_formats}"
        )

    if not isinstance(imgs, np.ndarray):
        imgs = np.array(imgs)

    formatted_heatmap_imgs = None
    heatmap_input_shape = None

    if input_format == "NCTHW":
        current_clip_len = clip_len
        if isinstance(clip_len, dict):
            current_clip_len = clip_len["RGB"]

        imgs = imgs.reshape((-1, num_clips, current_clip_len) + imgs.shape[1:])
        imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
        imgs = imgs.reshape((-1,) + imgs.shape[2:])

        if heatmap_imgs is not None:
            heatmap_clip_len = clip_len
            if isinstance(clip_len, dict):
                heatmap_clip_len = clip_len["Pose"]

            formatted_heatmap_imgs = heatmap_imgs.reshape(
                (-1, num_clips, heatmap_clip_len) + heatmap_imgs.shape[1:]
            )
            formatted_heatmap_imgs = np.transpose(
                formatted_heatmap_imgs, (0, 1, 3, 2, 4, 5)
            )
            formatted_heatmap_imgs = formatted_heatmap_imgs.reshape(
                (-1,) + formatted_heatmap_imgs.shape[2:]
            )
            heatmap_input_shape = formatted_heatmap_imgs.shape

    elif input_format == "NCTHW_Heatmap":
        # Similar to NCTHW but specifically for heatmap data
        imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
        # N_crops x N_clips x T x C x H x W
        imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
        # N_crops x N_clips x C x T x H x W
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        # M' x C x T x H x W where M' = N_crops x N_clips

    input_shape = imgs.shape
    return imgs, input_shape, formatted_heatmap_imgs, heatmap_input_shape


def format_shape_from_data(data: Dict, input_format: str) -> Dict:
    """
    Apply shape formatting to data dictionary (MMAction2 compatible interface).

    Args:
        data (Dict): Data dictionary containing:
            - imgs (np.ndarray): Input images array
            - num_clips (int): Number of clips
            - clip_len (int | Dict[str, int]): Number of frames in each clip
            - heatmap_imgs (np.ndarray, optional): Heatmap images array
            - modality (str) : The modality in which the models run (pose, RBG, etc...)

        input_format (str): Define the final data format

    Returns:
        Dict: Updated data dictionary with modified keys:
            - imgs: Formatted images array
            - input_shape: Shape of formatted images
            - heatmap_imgs: Formatted heatmap images (if originally present)
            - heatmap_input_shape: Shape of formatted heatmap images (if originally present)

    Example:
        >>> data = {
        ...     "imgs": np.random.rand(120, 64, 64, 17),  # T x H x W x C
        ...     "num_clips": 10,
        ...     "clip_len": 12
        ... }
        >>> result = format_shape_from_data(data, input_format="NCTHW")
        >>> print(result["imgs"].shape)  # (10, 17, 12, 64, 64) -> N_clips x C x T x H x W
    """
    imgs = data["imgs"]
    num_clips = data["num_clips"]
    clip_len = data["clip_len"]

    heatmap_imgs = data.get("heatmap_imgs", None)
    modality = data.get("modality", None)

    formatted_imgs, input_shape, formatted_heatmap_imgs, heatmap_input_shape = (
        _format_shape(
            imgs=imgs,
            input_format=input_format,
            num_clips=num_clips,
            clip_len=clip_len,
            heatmap_imgs=heatmap_imgs,
            modality=modality,
        )
    )

    data["imgs"] = formatted_imgs
    data["input_shape"] = input_shape

    if formatted_heatmap_imgs is not None:
        data["heatmap_imgs"] = formatted_heatmap_imgs
        data["heatmap_input_shape"] = heatmap_input_shape

    return data


def _handle_collect_keys(
    data: Dict,
    collect_keys: Optional[Tuple[str, ...]],
    packed_results: Dict,
) -> None:
    packed_results["inputs"] = {}
    for key in collect_keys:
        if key in data:
            packed_results["inputs"][key] = (
                np.array(data[key])
                if not isinstance(data[key], np.ndarray)
                else data[key]
            )


def _handle_label(data: Dict, data_sample: Dict) -> None:
    if "label" in data:
        label = data["label"]
        if isinstance(label, (list, tuple)):
            label = np.array(label, dtype=np.int64)
        elif isinstance(label, (int, float)):
            label = np.array([label], dtype=np.int64)
        elif isinstance(label, np.ndarray):
            if label.ndim == 0:
                label = np.array([label.item()], dtype=np.int64)
            else:
                label = label.astype(np.int64)
        else:
            label = np.array(label, dtype=np.int64)
        data_sample["gt_label"] = label


def _pack_action_inputs(
    data: Dict,
    collect_keys: Optional[Tuple[str, ...]] = None,
    meta_keys: Sequence[str] = ("img_shape", "img_key", "video_id", "timestamp"),
    mapping_table: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Pack the inputs data for action recognition models.

    Args:
        data (Dict): Input data dictionary containing arrays and metadata
        collect_keys (tuple[str], optional): The keys to be collected
            to packed_results['inputs']. If None, uses 'imgs' by default.
        meta_keys (Sequence[str]): The meta keys to be saved in the
            metainfo. Defaults to ("img_shape", "img_key", "video_id", "timestamp").
        mapping_table (Dict[str, str], optional): Mapping table for key names.
            Defaults to {"gt_bboxes": "bboxes", "gt_labels": "labels"}.

    Returns:
        Dict: Packed results dictionary with:
            - inputs: Input data arrays (as numpy arrays for framework-agnostic use)
            - data_samples: Dictionary containing labels, metadata, and algorithm keys
    """

    if mapping_table is None:
        mapping_table = {"gt_bboxes": "bboxes", "gt_labels": "labels"}

    packed_results = {}
    if collect_keys is not None:
        _handle_collect_keys(data, collect_keys, packed_results)
    else:
        if "imgs" in data:
            imgs = data["imgs"]
            packed_results["inputs"] = (
                np.array(imgs) if not isinstance(imgs, np.ndarray) else imgs
            )

    data_sample = {}
    _handle_label(data, data_sample)

    metainfo = {k: data[k] for k in meta_keys if k in data}
    data_sample["metainfo"] = metainfo

    # Apply mapping table for any mapped keys
    for old_key, new_key in mapping_table.items():
        if old_key in data:
            data_sample[new_key] = data[old_key]

    packed_results["data_samples"] = data_sample
    return packed_results


def pack_action_inputs_from_data(
    data: Dict,
    collect_keys: Optional[Tuple[str, ...]] = None,
    meta_keys: Sequence[str] = ("img_shape", "img_key", "video_id", "timestamp"),
) -> Dict:
    """
    Apply action input packing to data dictionary (MMAction2 compatible interface).

    Args:
        data (Dict): Data dictionary containing all pipeline results
        collect_keys (tuple[str], optional): The keys to be collected to inputs.
                                           If None, uses 'imgs' by default.
        meta_keys (Sequence[str]): The meta keys to be saved.
                                 Defaults to ("img_shape", "img_key", "video_id", "timestamp").
    Returns:
        Dict: Packed results dictionary ready for model inference with:
            - inputs: Input data arrays
            - data_samples: Dictionary containing labels, metadata, and algorithm keys

    Example:
        >>> data = {
        ...     "imgs": np.random.rand(5, 17, 24, 64, 64),  # Formatted heatmaps
        ...     "label": 42,
        ...     "img_shape": (64, 64)
        ... }
        >>> result = pack_action_inputs_from_data(data)
        >>> print(result["inputs"].shape)  # (5, 17, 24, 64, 64)
        >>> print(result["data_samples"]["gt_label"])  # [42]
    """
    return _pack_action_inputs(
        data=data,
        collect_keys=collect_keys,
        meta_keys=meta_keys,
    )
