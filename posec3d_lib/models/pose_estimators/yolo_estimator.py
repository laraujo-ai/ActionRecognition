import numpy as np
import logging
from typing import  List
from collections import defaultdict
import cv2

from posec3d_lib.models.pose_estimators.base import BasePoseEstimator
from posec3d_lib.models.functions import non_max_suppression


logger = logging.getLogger(__name__)

class YoloPoseEstimator(BasePoseEstimator):
    """YOLOv8-pose model for human pose estimation in video frames.
    
    This class implements pose detection using YOLOv8-pose architecture via ONNX Runtime.
    It detects human poses by identifying 17 COCO-format keypoints per person along with
    bounding boxes and confidence scores. This serves as the first stage in the action
    recognition pipeline, providing pose sequences for downstream PoseC3D processing.
    
    The pose estimation workflow:
    1. Preprocess input frames (resize, normalize, format conversion)
    2. Run YOLO-pose inference to detect keypoints and bounding boxes
    3. Apply non-maximum suppression to filter detections
    4. Post-process outputs into standardized format for action recognition
    
    Attributes:
        model: ONNX Runtime inference session for YOLOv8-pose
        keypoint_conf_threshold: Minimum confidence threshold for keypoint visibility
        inputs_details: Model input specification from ONNX metadata
        INPUT_SIZE: Model input dimensions (batch, channels, height, width)
        INPUT_NAME: Name of the model's input tensor
    """
    def __init__(self, model_engine, keypoint_conf_threshold: int):
        """Initialize YOLOv8-pose estimator.
        
        Args:
            model_engine: ONNX Runtime inference session for YOLOv8-pose model
            keypoint_conf_threshold: Minimum confidence threshold (0-1) for considering
                keypoints as visible. Keypoints below this threshold are marked as invisible.
        """
        self.model = model_engine
        self.keypoint_conf_threshold = keypoint_conf_threshold
        
        # Model information
        self.inputs_details = self.model.get_inputs()
        self.INPUT_SIZE = self.inputs_details[0].shape
        self.INPUT_NAME = self.inputs_details[0].name

    def preprocess(self, data) -> np.ndarray:
        """Preprocess input frame for YOLO-pose inference.
        
        Transforms input frame to match model requirements: resizes to model input size,
        converts BGR to RGB, normalizes pixel values to [0,1], adds batch dimension,
        and reorders to NCHW format expected by ONNX models.
        
        Args:
            data: Input frame as numpy array in BGR format (H, W, 3)
            
        Returns:
            np.ndarray: Preprocessed frame tensor of shape (1, 3, H, W) with
                values normalized to [0,1] and in RGB format
        """
        input_height, input_width = self.INPUT_SIZE[2], self.INPUT_SIZE[3]
        frame = cv2.resize(data, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
        img = frame[:, :, ::-1]
        img = img/255.00
        img = np.asarray(img, dtype=np.float32)
        img = np.expand_dims(img,0)
        img = img.transpose(0,3,1,2)

        return img

    def predict(self, frame: np.ndarray) -> dict:
        """Run YOLO-pose inference on preprocessed frame.
        
        Performs pose detection on a single frame, applying non-maximum suppression
        to filter overlapping detections and formatting results for downstream processing.
        
        Args:
            frame: Preprocessed frame tensor from preprocess() method,
                shape (1, 3, H, W) with normalized pixel values
                
        Returns:
            dict: Pose detection results containing:
                - keypoints: Array of shape (num_persons, 17, 2) with (x,y) coordinates
                - keypoint_scores: Array of shape (num_persons, 17) with confidence scores
                - keypoints_visible: Array of shape (num_persons, 17) with visibility mask
                - bbox_scores: Array of shape (num_persons,) with bounding box confidence
                - bboxes: Array of shape (num_persons, 4) with bounding box coordinates
                Returns empty dict if no poses are detected.
                
        Raises:
            Exception: If model inference fails
        """
        try:
            output = self.model.run([], {self.INPUT_NAME: frame})[0] 
            boxes, conf_scores, keypt_vectors = non_max_suppression(output[0], self.keypoint_conf_threshold) # Only first of the batch
            num_persons = len(keypt_vectors)

            if keypt_vectors is None:
                return {}

            frame_info = defaultdict(
                keypoints=np.zeros((num_persons, 17, 2)),
                keypoint_scores=np.zeros((num_persons, 17)),
                keypoints_visible=np.zeros((num_persons, 17)),
                bbox_scores=np.zeros((num_persons)),
                bboxes=np.zeros((num_persons, 4)),
            )
            self.postprocess(boxes, conf_scores, keypt_vectors, frame_info, num_persons)
            return frame_info
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            return {}

    def postprocess(self, boxes, conf_scores, keypt_vectors, frame_info: dict, num_persons: int) -> None:
        """Post-process YOLO-pose raw outputs into standardized format.
        
        Converts raw model outputs (bounding boxes, confidence scores, keypoint vectors)
        into the structured format expected by the action recognition pipeline. Handles
        coordinate extraction, confidence thresholding, and visibility mask generation.
        
        Args:
            boxes: Bounding box coordinates for detected persons, shape (num_persons, 4)
            conf_scores: Confidence scores for bounding boxes, shape (num_persons,)
            keypt_vectors: Flattened keypoint data per person, shape (num_persons, 51)
                where 51 = 17 keypoints * 3 values (x, y, confidence)
            frame_info: Output dictionary to populate with processed results
            num_persons: Number of detected persons in the frame
        """
        for pid in range(num_persons):
            keypoints_flat = keypt_vectors[pid]  # Shape: (51,) for 17 keypoints * 3 values each
            person_keypoints = keypoints_flat.reshape(17, 3)[:, :2]  # Extract x,y coordinates
            keypoints_score = keypoints_flat.reshape(17, 3)[:, 2]    # Extract confidence scores
            
            box = boxes[pid]  
            box_conf = conf_scores[pid] 
            mask = (keypoints_score >= self.keypoint_conf_threshold).astype(np.float32)

            frame_info["keypoints"][pid] = person_keypoints
            frame_info["keypoints_visible"][pid] = mask
            frame_info["keypoint_scores"][pid] = keypoints_score
            frame_info["bbox_scores"][pid] = box_conf
            frame_info["bboxes"][pid] = box

    def inference_on_video(self, frame_paths: List[str]) -> List[dict]:
        """Run pose estimation on a sequence of video frames.
        
        Processes a list of frame image files through the complete pose estimation
        pipeline (preprocess → predict → postprocess) and returns results formatted
        for the action recognition pipeline.
        
        Args:
            frame_paths: List of file paths to frame images to process
            
        Returns:
            List[dict]: List of pose detection results, one per frame, where each
                dict contains keypoints, scores, visibility, and bounding box information
                in the format expected by PoseC3D preprocessing
        """
        ret = []

        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            preprocessed_img = self.preprocess(frame)
            frame_info = self.predict(preprocessed_img)
            ret.append(frame_info)

        return ret
