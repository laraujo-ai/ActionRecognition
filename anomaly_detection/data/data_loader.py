import os
import tempfile
from glob import glob
from typing import List, Tuple, Dict, Any
import numpy as np
import tqdm

import posec3d_lib.providers as prov
from posec3d_lib.utils.video_recording import extract_frames
from anomaly_detection.data.schemas import DatasetConfig


class VideoDataLoader:
    """Video data loader for pose-based anomaly detection training.

    This class orchestrates the complete data loading pipeline from raw videos
    to feature vectors suitable for anomaly detection training. It handles pose
    estimation, feature extraction, and data organization across train/validation/test
    splits while supporting configurable sample limits for development.

    The processing workflow:
    1. Load video files from structured dataset directories
    2. Extract frames using temporal sampling strategies
    3. Estimate human poses using YOLOv8-pose detection
    4. Extract feature representations using PoseC3D backbone
    5. Organize features by normal/anomalous labels for training

    Attributes:
        config: Dataset configuration including paths and category mappings
        verbose: Enable detailed logging and progress tracking
        pose_estimator: YOLOv8-pose model for keypoint detection
        feature_extractor: PoseC3D model for pose sequence encoding
        normalize_features: Whether to L2-normalize extracted feature vectors
    """

    def __init__(
        self, config: DatasetConfig, model_config: Dict[str, Any], verbose: bool = True
    ):
        """Initialize video data loader with model components.

        Args:
            config: Dataset configuration with paths and categories
            model_config: Model configuration containing paths and settings
            verbose: Enable progress tracking and detailed logging
        """
        self.config = config
        self.verbose = verbose
        self._setup_model_components(model_config)

    def _setup_model_components(self, model_config: Dict[str, Any]) -> None:
        """Initialize pose estimation and feature extraction pipeline.

        Sets up the complete model pipeline including pose estimation, data
        preprocessing, and feature extraction components from configuration.

        Args:
            model_config: Dictionary containing model paths and preprocessing settings
        """
        config = prov.get_config(model_config["config_path"])
        self.pose_estimator = prov.get_yolo_pose_estimator(config)
        transforms_pipeline = prov.get_transformations_pipeline(config)
        preprocessor = prov.get_preprocessor(transforms_pipeline)
        onnx_model_engine = prov.get_model_engine(config)
        self.feature_extractor = prov.get_feature_extractor(
            preprocessor, onnx_model_engine
        )
        self.normalize_features = model_config.get("feature_normalization", True)

    def get_clips_path(self, directory: str, category: str) -> List[str]:
        """Retrieve video file paths for a specific action category.

        Searches for both MP4 and AVI video files within the specified category
        directory, supporting common video formats used in action recognition datasets.

        Args:
            directory: Path to dataset split directory (train/validation/test)
            category: Action category name (e.g., 'sitting_down', 'falling_down')

        Returns:
            List[str]: Absolute paths to all video files found in the category directory
        """
        mp4s = glob(os.path.join(directory, category, "*.mp4"))
        avis = glob(os.path.join(directory, category, "*.avi"))
        return mp4s + avis

    def get_all_clips_for_split(
        self, directory: str, categories: List[str], limit: int = -1
    ) -> List[str]:
        """Aggregate video clips across multiple categories for a dataset split.

        Collects video files from all specified categories within a dataset split
        directory, with optional limiting for rapid development and testing.

        Args:
            directory: Path to dataset split directory (train/validation/test)
            categories: List of action category names to include
            limit: Maximum number of clips to return (-1 for no limit)

        Returns:
            List[str]: Combined list of video file paths across all categories
        """
        clips = []
        for category in categories:
            clips.extend(self.get_clips_path(directory, category))

        if limit > 0:
            clips = clips[:limit]
        return clips

    def extract_feature_vector(self, video_path: str) -> np.ndarray:
        """Extract pose-based feature vector from video file.

        Processes a single video through the complete feature extraction pipeline:
        frame extraction, pose estimation, sequence preprocessing, and feature encoding.
        Handles temporary file management and optional feature normalization.

        Args:
            video_path: Absolute path to input video file

        Returns:
            np.ndarray: Feature vector of shape (feature_dim,) encoding the pose sequence.
                Optionally L2-normalized based on configuration settings.

        Raises:
            RuntimeError: If no frames can be extracted from the video file
        """
        tmp_dir = tempfile.TemporaryDirectory()
        try:
            frame_paths, frames, _ = extract_frames(video_path, tmp_dir.name)
            if len(frames) == 0:
                raise RuntimeError(f"No frames extracted from {video_path}")

            h, w, _ = frames[0].shape
            pose_results = self.pose_estimator.inference_on_video(frame_paths)
            data = {"pose_results": pose_results, "img_shape": (h, w)}
            feature_vector = self.feature_extractor.inference(data)

            if self.normalize_features:
                norm = np.linalg.norm(feature_vector)
                if norm > 0:
                    feature_vector = feature_vector / norm

            return feature_vector
        finally:
            tmp_dir.cleanup()

    def generate_feature_matrix(self, paths: List[str]) -> np.ndarray:
        """Extract features from multiple videos into a batch matrix.

        Processes a list of video files through feature extraction, handling errors
        gracefully and providing progress tracking. Combines individual feature vectors
        into a matrix suitable for machine learning training.

        Args:
            paths: List of absolute paths to video files to process

        Returns:
            np.ndarray: Feature matrix of shape (num_videos, feature_dim) where each
                row contains the feature vector for one video. Returns empty array
                if no videos can be processed successfully.
        """
        features = []
        failed_count = 0

        iterator = tqdm.tqdm(paths, desc="Processing videos") if self.verbose else paths

        for path in iterator:
            try:
                feature_vector = self.extract_feature_vector(path)
                features.append(feature_vector)
            except Exception as e:
                failed_count += 1
                if self.verbose:
                    print(f"Warning: Failed to extract features from {path}: {e}")

        if failed_count > 0 and self.verbose:
            print(f"Failed to process {failed_count}/{len(paths)} videos")

        if len(features) == 0:
            return np.zeros((0, 0))

        return np.stack(features, axis=0)

    def generate_labeled_split(
        self, normal_paths: List[str], anomalous_paths: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create labeled feature matrix from normal and anomalous video sets.

        Processes both normal and anomalous videos to create a combined feature matrix
        with corresponding binary labels for supervised anomaly detection training.

        Args:
            normal_paths: List of video paths containing normal behavior (label=0)
            anomalous_paths: List of video paths containing anomalous behavior (label=1)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Feature matrix of shape (total_videos, feature_dim)
                - Label array of shape (total_videos,) with 0=normal, 1=anomalous
        """
        if self.verbose:
            print(
                f"Processing {len(normal_paths)} normal and {len(anomalous_paths)} anomalous videos"
            )

        x_normal = self.generate_feature_matrix(normal_paths)
        x_anomalous = self.generate_feature_matrix(anomalous_paths)

        if x_normal.size == 0 and x_anomalous.size == 0:
            return np.zeros((0, 0)), np.zeros((0,), dtype=int)

        if x_normal.size > 0 and x_anomalous.size > 0:
            X = np.vstack([x_normal, x_anomalous])
        elif x_normal.size > 0:
            X = x_normal
        else:
            X = x_anomalous

        y_normal = (
            np.zeros((x_normal.shape[0],), dtype=int)
            if x_normal.size > 0
            else np.zeros((0,), dtype=int)
        )
        y_anomalous = (
            np.ones((x_anomalous.shape[0],), dtype=int)
            if x_anomalous.size > 0
            else np.zeros((0,), dtype=int)
        )
        y = np.concatenate([y_normal, y_anomalous])

        return X, y

    def load_dataset(
        self,
    ) -> Tuple[
        np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
    ]:
        """Load complete dataset for anomaly detection training.

        Orchestrates loading of all dataset splits (train/validation/test) with proper
        directory validation and feature extraction. Training data contains only normal
        samples for unsupervised learning, while validation and test splits contain
        both normal and anomalous samples for evaluation.

        Returns:
            Tuple containing:
                - X_train: Training features of shape (num_train, feature_dim) - normal only
                - (X_val, y_val): Validation features and labels for threshold tuning
                - (X_test, y_test): Test features and labels for final evaluation

        Raises:
            FileNotFoundError: If any required dataset split directory is missing
        """
        base_path = self.config.base_path

        train_dir = os.path.join(base_path, "train")
        val_dir = os.path.join(base_path, "validation")
        test_dir = os.path.join(base_path, "test")

        for split_name, split_dir in [
            ("train", train_dir),
            ("validation", val_dir),
            ("test", test_dir),
        ]:
            if not os.path.exists(split_dir):
                raise FileNotFoundError(
                    f"{split_name} directory not found: {split_dir}"
                )

        if self.verbose:
            print("Loading dataset splits...")

        train_normal_clips = self.get_all_clips_for_split(
            train_dir,
            self.config.normal_categories,
            self.config.limits.get("train_normal", -1),
        )
        X_train = self.generate_feature_matrix(train_normal_clips)

        val_normal_clips = self.get_all_clips_for_split(
            val_dir,
            self.config.normal_categories,
            self.config.limits.get("val_normal", -1),
        )
        val_anomalous_clips = self.get_all_clips_for_split(
            val_dir,
            self.config.anomaly_categories,
            self.config.limits.get("val_anomalous", -1),
        )
        x_val, y_val = self.generate_labeled_split(
            val_normal_clips, val_anomalous_clips
        )

        test_normal_clips = self.get_all_clips_for_split(
            test_dir,
            self.config.normal_categories,
            self.config.limits.get("test_normal", -1),
        )
        test_anomalous_clips = self.get_all_clips_for_split(
            test_dir,
            self.config.anomaly_categories,
            self.config.limits.get("test_anomalous", -1),
        )
        X_test, y_test = self.generate_labeled_split(
            test_normal_clips, test_anomalous_clips
        )

        if self.verbose:
            print(
                f"Dataset loaded - Train: {X_train.shape}, Validation: {x_val.shape}, Test: {X_test.shape}"
            )

        return X_train, (x_val, y_val), (X_test, y_test)
