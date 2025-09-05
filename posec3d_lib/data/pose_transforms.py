import numpy as np
from typing import Dict, Tuple, Optional, Union, List


def _get_test_clips(num_frames: int, clip_len: int, num_clips: int) -> np.ndarray:
    """Generate test clip indices - replicates MMAction2's _get_test_clips."""
    all_inds = []
    rng = np.random.default_rng(255)
    for i in range(num_clips):
        if num_frames < clip_len:
            start_ind = i if num_frames < num_clips else i * num_frames // num_clips
            inds = np.arange(start_ind, start_ind + clip_len)
        elif clip_len <= num_frames < clip_len * 2:
            basic = np.arange(clip_len)
            inds = rng.choice(clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = rng.integers(bsize)
            inds = bst + offset

        all_inds.append(inds)

    return np.concatenate(all_inds)


def _uniform_sample_frames(
    total_frames: int,
    clip_len: int,
    num_clips: int = 1,
    start_index: int = 0,
    keypoint: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Uniformly sample frame indices from a video in test mode.

    The function divides the video into `clip_len` segments of equal length and
    samples one frame from each segment.
    Args:
        total_frames (int): Total number of frames in the video
        clip_len (int): Number of frames in each clip
        num_clips (int): Number of clips to sample. Defaults to 1.
        start_index (int): Starting index offset. Defaults to 0.
        seed (int): Random seed for deterministic sampling. Defaults to 255.
        keypoint (np.ndarray, optional): Keypoint data with shape (num_person, num_frames, num_keypoints, 2).
                                       Used for transitional frame handling. Defaults to None.

    Returns:
        Tuple[np.ndarray, Dict]:
            - frame_inds: Array of sampled frame indices
            - metadata: Dictionary containing clip_len, num_clips, frame_interval

    """

    inds = _get_test_clips(total_frames, clip_len, num_clips)

    # Apply modulo to handle out-of-bounds indices
    inds = np.mod(inds, total_frames)
    inds = inds + start_index

    if keypoint is not None:
        assert (
            total_frames == keypoint.shape[1]
        ), f"Frame mismatch: total_frames={total_frames}, keypoint.shape[1]={keypoint.shape[1]}"

        num_person = keypoint.shape[0]
        num_persons = [num_person] * total_frames

        # Count actual number of persons per frame (non-zero keypoints)
        for i in range(total_frames):
            j = num_person - 1
            while j >= 0 and np.all(np.abs(keypoint[j, i]) < 1e-5):
                j -= 1
            num_persons[i] = j + 1

        # Mark transitional frames (where number of persons changes)
        transitional = [False] * total_frames
        for i in range(1, total_frames - 1):
            if num_persons[i] != num_persons[i - 1]:
                transitional[i] = transitional[i - 1] = True
            if num_persons[i] != num_persons[i + 1]:
                transitional[i] = transitional[i + 1] = True

        # Apply transitional frame handling
        inds_int = inds.astype(np.int64)
        coeff = np.array([transitional[i] for i in inds_int])
        inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

    metadata = {"clip_len": clip_len, "frame_interval": None, "num_clips": num_clips}
    return inds.astype(np.int32), metadata


def uniform_sample_frames_from_data(
    data: Dict, clip_len: int, num_clips: int = 1
) -> Dict:
    """
    Apply uniform frame sampling to data dictionary.

    Args:
        data (Dict): Data dictionary containing:
            - total_frames (int): Total number of frames
            - start_index (int, optional): Starting index offset. Defaults to 0.
            - keypoint (np.ndarray, optional): Keypoint data for transitional handling
        clip_len (int): Number of frames in each clip
        num_clips (int): Number of clips to sample. Defaults to 1.

    Returns:
        Dict: Updated data dictionary with added keys:
            - frame_inds: Array of sampled frame indices
            - clip_len: Number of frames per clip
            - frame_interval: Frame interval (always None for this transform)
            - num_clips: Number of clips

    Example:
        >>> data = {"total_frames": 100, "start_index": 0}
        >>> result = uniform_sample_frames_from_data(data, clip_len=48, num_clips=10)
        >>> print(result["frame_inds"].shape)  # (480,) for 10 clips * 48 frames
    """
    total_frames = data["total_frames"]
    start_index = data.get("start_index", 0)
    keypoint = data.get("keypoint", None)

    frame_inds, metadata = _uniform_sample_frames(
        total_frames=total_frames,
        clip_len=clip_len,
        num_clips=num_clips,
        start_index=start_index,
        keypoint=keypoint,
    )

    data["frame_inds"] = frame_inds
    data.update(metadata)

    return data


def _pose_decode(
    keypoint: np.ndarray,
    frame_inds: np.ndarray,
    offset: int = 0,
    keypoint_score: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load and decode pose keypoints with given frame indices.

    This function selects keypoint data for specific frame indices, which is typically used
    after frame sampling to extract only the relevant keypoint data.

    Args:
        keypoint (np.ndarray): Keypoint data with shape (num_person, num_frames, num_keypoints, 2)
        frame_inds (np.ndarray): Frame indices to select, shape (num_selected_frames,)
        offset (int): Index offset to add to frame_inds. Defaults to 0.
        keypoint_score (np.ndarray, optional): Keypoint confidence scores with shape
                                             (num_person, num_frames, num_keypoints). Defaults to None.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]:
            - keypoint: Selected keypoint data with shape (num_person, num_selected_frames, num_keypoints, 2)
            - keypoint_score: Selected keypoint scores (if provided) with shape (num_person, num_selected_frames, num_keypoints)
    """

    def _load_kp(kp: np.ndarray, frame_inds: np.ndarray) -> np.ndarray:
        """Load keypoints according to sampled indexes - replicates MMAction2's _load_kp."""
        return kp[:, frame_inds].astype(np.float32)

    def _load_kpscore(kpscore: np.ndarray, frame_inds: np.ndarray) -> np.ndarray:
        """Load keypoint scores according to sampled indexes - replicates MMAction2's _load_kpscore."""
        return kpscore[:, frame_inds].astype(np.float32)

    if frame_inds.ndim != 1:
        frame_inds = np.squeeze(frame_inds)

    final_frame_inds = frame_inds + offset
    max_frames = keypoint.shape[1]
    if np.any(final_frame_inds >= max_frames) or np.any(final_frame_inds < 0):
        raise IndexError(
            f"Frame indices {final_frame_inds} are out of bounds for keypoint array with {max_frames} frames"
        )
    decoded_keypoint = _load_kp(keypoint, final_frame_inds)
    decoded_keypoint_score = None
    if keypoint_score is not None:
        if keypoint_score.shape[1] != max_frames:
            raise ValueError(
                f"keypoint_score frames {keypoint_score.shape[1]} doesn't match keypoint frames {max_frames}"
            )
        decoded_keypoint_score = _load_kpscore(keypoint_score, final_frame_inds)

    return decoded_keypoint, decoded_keypoint_score


def pose_decode_from_data(data: Dict, offset: int = 0) -> Dict:
    """
    Apply pose decoding to data dictionary.

    Args:
        data (Dict): Data dictionary containing:
            - keypoint (np.ndarray): Keypoint data with shape (num_person, num_frames, num_keypoints, 2)
            - frame_inds (np.ndarray): Frame indices to select
            - keypoint_score (np.ndarray, optional): Keypoint confidence scores
            - total_frames (int, optional): Total number of frames (will be inferred if not provided)
        offset (int): Index offset to add to frame_inds. Defaults to 0.

    Returns:
        Dict: Updated data dictionary with modified keys:
            - keypoint: Selected keypoint data
            - keypoint_score: Selected keypoint scores (if originally present)
            - total_frames: Set to original total frames if not present

    Example:
        >>> data = {
        ...     "keypoint": np.random.rand(1, 100, 17, 2),
        ...     "frame_inds": np.array([0, 10, 20, 30, 40])
        ... }
        >>> result = pose_decode_from_data(data)
        >>> print(result["keypoint"].shape)  # (1, 5, 17, 2)
    """

    keypoint = data["keypoint"]
    frame_inds = data["frame_inds"]
    keypoint_score = data.get("keypoint_score", None)

    decoded_keypoint, decoded_keypoint_score = _pose_decode(
        keypoint=keypoint,
        frame_inds=frame_inds,
        offset=offset,
        keypoint_score=keypoint_score,
    )

    data["keypoint"] = decoded_keypoint
    if decoded_keypoint_score is not None:
        data["keypoint_score"] = decoded_keypoint_score

    return data


def _pair(value: Union[float, int, Tuple]) -> Tuple:
    """
    Convert a single value to a pair, or return the pair if already a tuple.
    This replicates the behavior of torch.nn.modules.utils._pair.

    Args:
        value: Single number or tuple of two numbers

    Returns:
        Tuple of two numbers
    """
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value, value)


def _combine_quadruple(a: Tuple, b: Tuple) -> Tuple:
    """
    Combine two crop quadruples.
    This replicates MMAction2's _combine_quadruple function.

    Args:
        a: First quadruple (x, y, w, h)
        b: Second quadruple (x, y, w, h)

    Returns:
        Combined quadruple
    """
    return a[0] + a[2] * b[0], a[1] + a[3] * b[1], a[2] * b[2], a[3] * b[3]


def _pose_compact(
    keypoint: np.ndarray,
    img_shape: Tuple[int, int],
    padding: float = 0.25,
    threshold: int = 10,
    hw_ratio: Optional[Union[float, Tuple[float]]] = None,
    allow_imgpad: bool = True,
    crop_quadruple: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, float, float, float]]:
    """
    Convert keypoint coordinates to make them more compact by finding a tight bounding box.

    This function finds a tight bounding box that surrounds all joints in each frame,
    then expands the tight box by a given padding ratio. For example, if
    'padding == 0.25', then the expanded box has unchanged center, and 1.25x
    width and height.

    Args:
        keypoint (np.ndarray): Keypoint data with shape (num_person, num_frames, num_keypoints, 2)
        img_shape (Tuple[int, int]): Original image shape (height, width)
        padding (float): The padding size. Defaults to 0.25.
        threshold (int): The threshold for the tight bounding box. If the width
                        or height of the tight bounding box is smaller than the threshold,
                        we do not perform the compact operation. Defaults to 10.
        hw_ratio (float | tuple[float] | None): The hw_ratio of the expanded
                                              box. Float indicates the specific ratio and tuple indicates a
                                              ratio range. If set as None, it means there is no requirement on
                                              hw_ratio. Defaults to None.
        allow_imgpad (bool): Whether to allow expanding the box outside the
                           image to meet the hw_ratio requirement. Defaults to True.
        crop_quadruple (Tuple[float, float, float, float], optional): Previous crop info (x, y, w, h)
                                                                     in normalized coordinates. Defaults to None.

    Returns:
        Tuple[np.ndarray, Tuple[int, int], Tuple[float, float, float, float]]:
            - keypoint: Adjusted keypoint coordinates
            - new_img_shape: New image shape after compacting
            - crop_quadruple: Crop information for this operation

    """

    h, w = img_shape
    kp = keypoint.copy()

    kp[np.isnan(kp)] = 0.0
    kp_x = kp[..., 0]
    kp_y = kp[..., 1]

    valid_x = kp_x[kp_x != 0]
    valid_y = kp_y[kp_y != 0]

    if len(valid_x) == 0 or len(valid_y) == 0:
        final_crop_quadruple = (
            crop_quadruple if crop_quadruple is not None else (0.0, 0.0, 1.0, 1.0)
        )
        return kp, img_shape, final_crop_quadruple

    min_x = np.min(valid_x)
    min_y = np.min(valid_y)
    max_x = np.max(valid_x)
    max_y = np.max(valid_y)

    if max_x - min_x < threshold or max_y - min_y < threshold:
        final_crop_quadruple = (
            crop_quadruple if crop_quadruple is not None else (0.0, 0.0, 1.0, 1.0)
        )
        return kp, img_shape, final_crop_quadruple

    center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
    half_width = (max_x - min_x) / 2 * (1 + padding)
    half_height = (max_y - min_y) / 2 * (1 + padding)

    min_x, max_x = center[0] - half_width, center[0] + half_width
    min_y, max_y = center[1] - half_height, center[1] + half_height

    min_x, min_y = int(min_x), int(min_y)
    max_x, max_y = int(max_x), int(max_y)

    # Adjust keypoint coordinates to new reference frame
    kp_x[kp_x != 0] -= min_x
    kp_y[kp_y != 0] -= min_y

    new_shape = (max_y - min_y, max_x - min_x)

    # Calculate crop quadruple in normalized coordinates
    new_crop_quadruple = (
        min_x / w,
        min_y / h,
        (max_x - min_x) / w,
        (max_y - min_y) / h,
    )

    # Combine with previous crop quadruple if provided
    if crop_quadruple is not None:
        final_crop_quadruple = _combine_quadruple(crop_quadruple, new_crop_quadruple)
    else:
        # Default crop quadruple (no previous cropping)
        base_quadruple = (0.0, 0.0, 1.0, 1.0)
        final_crop_quadruple = _combine_quadruple(base_quadruple, new_crop_quadruple)

    return kp, new_shape, final_crop_quadruple


def pose_compact_from_data(
    data: Dict,
    padding: float = 0.25,
    threshold: int = 10,
    hw_ratio: Optional[Union[float, Tuple[float]]] = None,
    allow_imgpad: bool = True,
) -> Dict:
    """
    Apply pose compacting to data dictionary.

    Args:
        data (Dict): Data dictionary containing:
            - keypoint (np.ndarray): Keypoint data with shape (num_person, num_frames, num_keypoints, 2)
            - img_shape (Tuple[int, int]): Image shape (height, width)
            - crop_quadruple (Tuple[float, float, float, float], optional): Previous crop info
        padding (float): The padding size. Defaults to 0.25.
        threshold (int): The threshold for the tight bounding box. Defaults to 10.
        hw_ratio (float | tuple[float] | None): The hw_ratio constraint. Defaults to None.
        allow_imgpad (bool): Whether to allow expanding outside image bounds. Defaults to True.

    Returns:
        Dict: Updated data dictionary with modified keys:
            - keypoint: Adjusted keypoint coordinates
            - img_shape: New image shape after compacting
            - crop_quadruple: Updated crop information

    Example:
        >>> data = {
        ...     "keypoint": np.random.rand(1, 50, 17, 2) * 100,
        ...     "img_shape": (480, 640)
        ... }
        >>> result = pose_compact_from_data(data)
        >>> print(result["img_shape"])  # New compacted dimensions
    """
    keypoint = data["keypoint"]
    img_shape = data["img_shape"]
    crop_quadruple = data.get("crop_quadruple", None)

    compacted_keypoint, new_img_shape, new_crop_quadruple = _pose_compact(
        keypoint=keypoint,
        img_shape=img_shape,
        padding=padding,
        threshold=threshold,
        hw_ratio=hw_ratio,
        allow_imgpad=allow_imgpad,
        crop_quadruple=crop_quadruple,
    )

    data["keypoint"] = compacted_keypoint
    data["img_shape"] = new_img_shape
    data["crop_quadruple"] = new_crop_quadruple

    return data


def generate_keypoint_heatmap_test_mode(
    keypoint: np.ndarray,
    img_shape: Tuple[int, int],
    sigma: float = 0.6,
    use_score: bool = True,
    scaling: float = 1.0,
    keypoint_score: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate pseudo heatmaps based on joint coordinates and confidence scores.

    Args:
        keypoint (np.ndarray): Keypoint data with shape (num_person, num_frames, num_keypoints, 2)
        img_shape (Tuple[int, int]): Image shape (height, width)
        sigma (float): The sigma of the generated gaussian map. Defaults to 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
                         of the gaussian maps. Defaults to True.
        scaling (float): The ratio to scale the heatmaps. Defaults to 1.0.
        keypoint_score (np.ndarray, optional): Keypoint confidence scores with shape
                                             (num_person, num_frames, num_keypoints). Defaults to None.

    Returns:
        np.ndarray: Generated pseudo heatmaps with shape (num_frames, num_channels, img_h, img_w)
                   where num_channels = num_keypoints (if with_kp) + num_limbs (if with_limb)

    """

    def generate_single_keypoint_heatmap(
        heatmap: np.ndarray,
        centers: np.ndarray,
        max_values: np.ndarray,
        sigma: float,
        eps: float = 1e-4,
    ) -> None:
        """Generate pseudo heatmap for one keypoint in one frame."""
        img_h, img_w = heatmap.shape

        for center, max_value in zip(centers, max_values):
            if max_value < eps:
                continue

            mu_x, mu_y = center[0], center[1]
            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)

            x = np.arange(st_x, ed_x, 1, dtype=np.float32)
            y = np.arange(st_y, ed_y, 1, dtype=np.float32)

            # If the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma**2)
            patch = patch * max_value
            heatmap[st_y:ed_y, st_x:ed_x] = np.maximum(
                heatmap[st_y:ed_y, st_x:ed_x], patch
            )

    def generate_frame_heatmaps(
        heatmaps: np.ndarray,
        kps: np.ndarray,
        max_values: np.ndarray,
        sigma: float,
    ) -> None:
        """Generate pseudo heatmap for all keypoints and limbs in one frame."""
        channel_idx = 0
        num_kp = kps.shape[1]
        for i in range(num_kp):
            generate_single_keypoint_heatmap(
                heatmaps[channel_idx], kps[:, i], max_values[:, i], sigma
            )
            channel_idx += 1

    all_kps = keypoint.astype(np.float32)
    kp_shape = all_kps.shape

    if keypoint_score is not None:
        all_kpscores = keypoint_score.astype(np.float32)

    img_h, img_w = img_shape

    # Scale img_h, img_w and keypoints
    img_h = int(img_h * scaling + 0.5)
    img_w = int(img_w * scaling + 0.5)
    all_kps[..., :2] *= scaling

    num_frame = kp_shape[1]
    num_c = 0
    num_c += all_kps.shape[2]

    ret = np.zeros([num_frame, num_c, img_h, img_w], dtype=np.float32)

    for i in range(num_frame):
        kps = all_kps[:, i]
        kpscores = all_kpscores[:, i] if use_score else np.ones_like(all_kpscores[:, i])
        generate_frame_heatmaps(ret[i], kps, kpscores, sigma)

    return ret


def generate_pose_target_from_data(
    data: Dict,
    sigma: float = 0.6,
    use_score: bool = True,
    scaling: float = 1.0,
) -> Dict:
    """
    Apply pose target generation to data dictionary.

    Args:
        data (Dict): Data dictionary containing:
            - keypoint (np.ndarray): Keypoint data with shape (num_person, num_frames, num_keypoints, 2)
            - img_shape (Tuple[int, int]): Image shape (height, width)
            - keypoint_score (np.ndarray, optional): Keypoint confidence scores
        sigma (float): The sigma of the generated gaussian map. Defaults to 0.6.
        use_score (bool): Use the confidence score of keypoints. Defaults to True.
        scaling (float): The ratio to scale the heatmaps. Defaults to 1.0.

    Returns:
        Dict: Updated data dictionary with added key:
            - imgs: Generated pseudo heatmaps (if "imgs" not already present)
            - heatmap_imgs: Generated pseudo heatmaps (if "imgs" already present)

    Example:
        >>> data = {
        ...     "keypoint": np.random.rand(1, 50, 17, 2) * 100,
        ...     "img_shape": (64, 64)
        ... }
        >>> result = generate_pose_target_from_data(data, sigma=0.6)
        >>> print(result["imgs"].shape)  # (50, 17, 64, 64) for keypoint heatmaps
    """
    keypoint = data["keypoint"]
    img_shape = data["img_shape"]
    keypoint_score = data.get("keypoint_score", None)

    heatmaps = generate_keypoint_heatmap_test_mode(
        keypoint=keypoint,
        img_shape=img_shape,
        sigma=sigma,
        use_score=use_score,
        scaling=scaling,
        keypoint_score=keypoint_score,
    )

    data["imgs"] = heatmaps
    return data
