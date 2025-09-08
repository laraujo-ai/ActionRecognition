import cv2
import numpy as np
import logging
from typing import Tuple, List
from collections import defaultdict

from posec3d_lib.models.pose_estimators.base import BasePoseEstimator

logger = logging.getLogger(__name__)


class YoloPoseEstimator(BasePoseEstimator):
    def __init__(self, model_engine, keypoint_conf_threshold: int):
        self.model = model_engine
        self.keypoint_conf_threshold = keypoint_conf_threshold

    def preprocess(self, data) -> None: ...
    def predict(self, frame_path: str) -> dict:
        """
        Runs Yolo pose estimation on a single given frame.
        """
        try:
            pose_results = self.model(frame_path)[0]
            num_persons = len(pose_results.keypoints)

            if not (
                hasattr(pose_results, "keypoints")
                and pose_results.keypoints is not None
            ):
                return {}

            frame_info = defaultdict(
                keypoints=np.zeros((num_persons, 17, 2)),
                keypoint_scores=np.zeros((num_persons, 17)),
                keypoints_visible=np.zeros((num_persons, 17)),
                bbox_scores=np.zeros((num_persons)),
                bboxes=np.zeros((num_persons, 4)),
            )
            self.postprocess(pose_results, frame_info, num_persons)
            return frame_info
        except Exception as e:
            logger.error(f"Pose estimation failed for {frame_path}: {e}")
            return {}

    def postprocess(self, pose_results, frame_info: dict, num_persons: int) -> None:
        """
        Post processes a single pose estimation. This postprocessing code is designed so that the frame_info
        fits exactly into the rest of our pipeline.
        """

        for pid in range(num_persons):
            person_keypoints = pose_results.keypoints.xy.cpu().numpy()[pid]
            keypoints_score = pose_results.keypoints.conf.cpu().numpy()[pid]
            box = pose_results.boxes.xyxy.cpu().numpy()[pid]
            box_conf = pose_results.boxes.conf.cpu().numpy()[pid]
            mask = (keypoints_score >= self.keypoint_conf_threshold).astype(np.float32)

            frame_info["keypoints"][pid] = person_keypoints
            frame_info["keypoints_visible"][pid] = mask
            frame_info["keypoint_scores"][pid] = keypoints_score
            frame_info["bbox_scores"][pid] = box_conf
            frame_info["bboxes"][pid] = box

    def inference_on_video(self, frame_paths: List[str]) -> List[dict]:
        """
        Run YOLO-pose on frames and return list of dicts formatted like our action recognition
        pipeline expects.
        """
        ret = []

        for frame_path in frame_paths:
            frame_info = self.predict(frame_path)
            ret.append(frame_info)

        return ret
