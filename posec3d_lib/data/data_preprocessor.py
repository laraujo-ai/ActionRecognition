import numpy as np
from typing import Dict, Union, Tuple, Optional, Sequence, List, Any


def stack_batch_numpy(inputs: List[np.ndarray]) -> np.ndarray:
    """
    Stack a list of numpy arrays into a batch.
    Args:
        inputs (List[np.ndarray]): List of input arrays to stack

    Returns:
        np.ndarray: Stacked batch array
    """
    if not inputs:
        raise ValueError("Cannot stack empty list of inputs")

    first_shape = inputs[0].shape
    for i, inp in enumerate(inputs):
        if inp.shape != first_shape:
            raise ValueError(f"Input {i} has shape {inp.shape}, expected {first_shape}")

    return np.stack(inputs, axis=0)


def action_data_preprocess_test_mode(
    inputs: Union[np.ndarray, List[np.ndarray]],
    data_samples: Optional[List[Dict]] = None,
    to_float32: bool = True,
) -> Tuple[np.ndarray, Optional[List[Dict]]]:
    """
    Preprocess action data for model inference.
    Args:
        inputs (np.ndarray | List[np.ndarray]): Input data arrays
        data_samples (List[Dict], optional): Data sample metadata. Defaults to None.
        to_float32 (bool): Whether to convert data to float32. Defaults to True.

    Returns:
        Tuple[np.ndarray, Optional[List[Dict]]]:
            - batch_inputs: Preprocessed batch inputs ready for model
            - data_samples: Processed data samples metadata

    """

    if isinstance(inputs, list):
        batch_inputs = stack_batch_numpy(inputs)
    else:
        raise TypeError(f"Unsupported inputs type: {type(inputs)}")

    if to_float32:
        batch_inputs = batch_inputs.astype(np.float32)

    return batch_inputs, data_samples


def action_data_preprocess_from_packed_data(
    packed_data: Dict,
    to_float32: bool = True,
) -> Dict:
    """
    Apply action data preprocessing to packed data (MMAction2 compatible interface).

    Args:
        packed_data (Dict): Packed data dictionary from PackActionInputs containing:
            - inputs: Input data arrays
            - data_samples: Data sample metadata
        mean (Sequence[float | int], optional): The pixel mean of channels. Defaults to None.
        std (Sequence[float | int], optional): The pixel standard deviation of channels. Defaults to None.
        to_rgb (bool): Whether to convert from BGR to RGB. Defaults to False.
        to_float32 (bool): Whether to convert data to float32. Defaults to True.
        format_shape (str): Format shape of input data. Defaults to "NCHW".

    Returns:
        Dict: Preprocessed data dictionary ready for model inference with:
            - inputs: Preprocessed batch inputs
            - data_samples: Processed data samples metadata

    Example:
        >>> packed_data = {
        ...     "inputs": np.random.rand(10, 17, 48, 64, 64),
        ...     "data_samples": {"gt_label": np.array([15]), "metainfo": {...}}
        ... }
        >>> result = action_data_preprocess_from_packed_data(
        ...     packed_data,
        ...     mean=[0.485, 0.456, 0.406],
        ...     std=[0.229, 0.224, 0.225],
        ...     format_shape="NCTHW"
        ... )
        >>> print(result["inputs"].shape)  # Same shape but normalized
    """
    inputs = packed_data["inputs"]
    data_samples = packed_data.get("data_samples", None)
    processed_inputs, processed_data_samples = action_data_preprocess_test_mode(
        inputs=inputs,
        data_samples=[data_samples] if data_samples is not None else None,
        to_float32=to_float32,
    )

    result = packed_data.copy()
    result["inputs"] = processed_inputs
    if processed_data_samples is not None:
        result["data_samples"] = processed_data_samples[0]

    return result
