from typing import Dict
import numpy as np
from collections import defaultdict

from posec3d_lib.models.preprocessors.base import BasePreprocessor
from posec3d_lib.utils.composer import Posec3dComposer
from posec3d_lib.data.data_preprocessor import action_data_preprocess_from_packed_data
from posec3d_lib.data.processing import pseudo_collate


class Posec3dPreprocessor(BasePreprocessor):
    """Preprocessor for PoseC3D models that transforms pose sequences into model input format.
    
    This class converts pose detection results into the format expected by PoseC3D models.
    It handles temporal sequences of human pose keypoints and transforms them into
    3D heatmap volumes suitable for action recognition or feature extraction.
    
    The preprocessing pipeline:
    1. Aggregates pose keypoints across frames and persons
    2. Applies transformation pipeline (sampling, normalization, etc.)
    3. Formats data for ONNX model input
    
    Attributes:
        transforms_pipeline: Posec3dComposer containing transformation steps
    """
    def __init__(self, transforms_pipeline: Posec3dComposer):
        """Initialize PoseC3D preprocessor.
        
        Args:
            transforms_pipeline: Posec3dComposer containing the sequence of
                transformations to apply to pose data
        """
        self.transforms_pipeline = transforms_pipeline

    def _prepare_input_for_pipeline(self, pose_results: list, h: int, w: int) -> dict:
        """Prepare pose detection results for the transformation pipeline.
        
        Converts list of per-frame pose results into the standardized format
        expected by the PoseC3D transformation pipeline. Creates fake annotation
        structure and aggregates keypoints across frames and persons.
        
        Args:
            pose_results: List of pose detection results per frame, where each
                frame contains keypoints, scores, and visibility information
            h: Height of the original video frames
            w: Width of the original video frames
            
        Returns:
            dict: Formatted data structure containing:
                - keypoint: Array of shape (num_person, num_frame, num_keypoint, 2)
                - keypoint_score: Array of shape (num_person, num_frame, num_keypoint)
                - img_shape: Original image dimensions
                - total_frames: Number of frames in the sequence
        """
        num_keypoint = pose_results[0]["keypoints"].shape[1]
        num_frame = len(pose_results)
        num_person = max([len(x["keypoints"]) for x in pose_results])
        fake_anno = defaultdict(
            frame_dict="",
            label=-1,
            img_shape=(h, w),
            origin_shape=(h, w),
            start_index=0,
            modality="Pose",
            total_frames=num_frame,
        )

        keypoint = np.zeros((num_frame, num_person, num_keypoint, 2), dtype=np.float16)
        keypoint_score = np.zeros(
            (num_frame, num_person, num_keypoint), dtype=np.float16
        )

        for f_idx, frm_pose in enumerate(pose_results):
            frm_num_persons = frm_pose["keypoints"].shape[0]
            for p_idx in range(frm_num_persons):
                keypoint[f_idx, p_idx] = frm_pose["keypoints"][p_idx]
                keypoint_score[f_idx, p_idx] = frm_pose["keypoint_scores"][p_idx]

        fake_anno["keypoint"] = keypoint.transpose((1, 0, 2, 3))
        fake_anno["keypoint_score"] = keypoint_score.transpose((1, 0, 2))

        return fake_anno

    def process(self, data) -> np.ndarray:
        """Process pose data through the complete preprocessing pipeline.
        
        Takes raw pose detection results and transforms them into the input format
        expected by PoseC3D ONNX models. Applies all transformation steps including
        temporal sampling, pose normalization, and heatmap generation.
        
        Args:
            data: Dictionary containing:
                - pose_results: List of pose detections per frame
                - img_shape: Tuple of (height, width) for the video frames
                
        Returns:
            np.ndarray: Preprocessed input tensor ready for model inference,
                typically of shape (batch, channels, time, height, width)
        """

        h, w = data["img_shape"]
        pose_results = data["pose_results"]
        data = self._prepare_input_for_pipeline(pose_results, h, w)

        data = self.transforms_pipeline(data)
        data = pseudo_collate([data])

        preprocessed_data = action_data_preprocess_from_packed_data(data)
        inputs = np.reshape(
            preprocessed_data["inputs"], (-1,) + preprocessed_data["inputs"].shape[2:]
        )
        return inputs
