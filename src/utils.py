from anomaly_detection.utils.functions import compute_mahalanobis_scores
import numpy as np


def handle_anomaly_inference(model, input_: np.ndarray):

    input_scaled = model.scaler.transform(input_)
    input_proc = (
        model.pca.transform(input_scaled) if model.pca is not None else input_scaled
    )

    scores = compute_mahalanobis_scores(input_proc, model.mean, model.inv_covariance)
    predictions = (scores > model.threshold).astype(int)
    class_map = {0: "Not an Anomaly", 1: "Anomaly Detected"}

    return class_map[predictions[0]]  # Only one clip per time


def is_rtsp_stream(media_link: str) -> bool:
    """Check if media link is an RTSP stream URL.

    Args:
        media_link: URL or file path to check

    Returns:
        bool: True if media_link is an RTSP stream, False otherwise
    """
    return isinstance(media_link, str) and media_link.startswith("rtsp://")
