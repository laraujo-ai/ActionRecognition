from typing import Dict
import numpy as np
from collections import defaultdict

from models.preprocessors.base import BasePreprocessor
from utils.composer import Posec3dComposer
from data.data_preprocessor import action_data_preprocess_from_packed_data
from data.processing import pseudo_collate


class Posec3dPreprocessor(BasePreprocessor):
    def __init__(self, transforms_pipeline: Posec3dComposer):
        self.transforms_pipeline = transforms_pipeline

    def _prepare_input_for_pipeline(self, pose_results: list, h: int, w: int) -> dict:
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
