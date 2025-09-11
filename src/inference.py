import numpy as np
import logging
import gc
from queue import Queue
from typing import Any
from src.utils import handle_anomaly_inference

logger = logging.getLogger(__name__)


def inference_thread(
    pose_estimator: Any,
    action_model: Any,
    anomaly_detector: Any,
    results_queue: Queue,
    clips_queue: Queue,
) -> None:
    """Process video clips through pose estimation and action recognition pipeline.

    Continuously processes clips from the clips_queue by:
    1. Extracting pose keypoints from video frames using YOLO pose estimator
    2. Running action recognition on pose data using PoseC3D model
    3. Publishing results to results_queue

    This function is designed to run in a separate thread and handles its own
    error recovery and memory management.

    Args:
        pose_estimator: YOLO pose estimation model that processes frame paths
        action_model: PoseC3D action recognition model that processes pose data
        results_queue: Queue to publish inference results to
        clips_queue: Queue to consume video clips from

    Returns:
        None: Function runs until poison pill (None) is received or critical error occurs

    Note:
        - Thread-safe function designed for concurrent execution
        - Performs aggressive memory cleanup to prevent leaks
        - Validates input data and handles missing files gracefully
        - Uses timeout on queue operations to prevent indefinite blocking
    """

    while True:
        try:
            clip = clips_queue.get(timeout=1)
            if clip is None:  # Poison pill to stop thread
                break

            if clip.clip_size <= 0:
                continue

            pose_results = pose_estimator.inference_on_video(clip.frame_paths)
            h, w = clip.frame_shape

            if len(pose_results) == 0:
                continue

            data = {"pose_results": pose_results, "img_shape": (h, w)}

            inference_results = action_model.inference(data)
            assert isinstance(inference_results, str) or isinstance(
                inference_results, np.ndarray
            )
            if anomaly_detector:
                inference_results = handle_anomaly_inference(model, inference_results)

            results_queue.put(inference_results)

            # Memory cleanup
            del clip, pose_results, data
            gc.collect()

        except Exception as e:
            from queue import Empty as QueueEmpty

            if isinstance(e, QueueEmpty):
                logger.debug("Queue timeout, continuing...")
                continue
            else:
                logger.error(f"Inference error: {e}")
                continue
