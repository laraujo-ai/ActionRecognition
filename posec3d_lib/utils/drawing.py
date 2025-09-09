import cv2
import numpy as np
from typing import Optional, Sequence, Tuple


DRAW_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to draw keypoints and connections
# COCO-format skeleton connections for 17 keypoints
# Defines which keypoints should be connected to form the human skeleton
COCO_CONNECTIONS = [
    (0, 1),   # nose -> left_eye
    (0, 2),   # nose -> right_eye
    (1, 3),   # left_eye -> left_ear
    (2, 4),   # right_eye -> right_ear
    (5, 6),   # left_shoulder -> right_shoulder
    (5, 7),   # left_shoulder -> left_elbow
    (7, 9),   # left_elbow -> left_wrist
    (6, 8),   # right_shoulder -> right_elbow
    (8, 10),  # right_elbow -> right_wrist
    (5, 11),  # left_shoulder -> left_hip
    (6, 12),  # right_shoulder -> right_hip
    (11, 12), # left_hip -> right_hip
    (11, 13), # left_hip -> left_knee
    (13, 15), # left_knee -> left_ankle
    (12, 14), # right_hip -> right_knee
    (14, 16), # right_knee -> right_ankle
]
DEFAULT_ACTION_RECOGNITION_SPOT = (20, 30)  # Default position for action text overlay


def draw_on_video(
    frames: Sequence[np.ndarray], pose_results: list, recognition_result: str
) -> Sequence[np.ndarray]:
    """Draw pose skeletons and action recognition results on video frames.
    
    Processes a sequence of video frames by overlaying pose estimation results
    (keypoints, skeleton connections, bounding boxes) and action recognition
    text. This is the main entry point for video visualization.
    
    Args:
        frames: Sequence of video frames as numpy arrays (H, W, 3)
        pose_results: List of pose detection results per frame, where each
            result contains keypoints, scores, and bounding box information
        recognition_result: Predicted action class name to display as text
        
    Returns:
        Sequence[np.ndarray]: Processed frames with pose and action overlays
    """
    frames_ready = []
    for frame, pose_result in zip(frames, pose_results):
        draw_skeleton(frame, pose_result, COCO_CONNECTIONS)
        draw_recognition_result(frame, recognition_result)
        frames_ready.append(frame)

    return frames_ready


def draw_recognition_result(frame: np.ndarray, recognition_result: str) -> None:
    """Draw action recognition result text on a video frame.
    
    Overlays the predicted action class name as text in the upper-left corner
    of the frame using OpenCV text rendering with predefined styling.
    
    Args:
        frame: Input video frame to draw on (modified in-place)
        recognition_result: Action class name to display as text overlay
    """
    cv2.putText(
        frame,
        recognition_result,
        DEFAULT_ACTION_RECOGNITION_SPOT,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 55, 55),
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def draw_skeleton(
    img: np.ndarray, pose_result: dict, connections: Sequence[Tuple[int, int]]
) -> None:
    """Draw pose skeleton with keypoints, connections, and bounding boxes on image.
    
    Renders a complete pose visualization including:
    1. Keypoint circles (red) for joints above confidence threshold
    2. Skeleton connections (green lines) between valid keypoint pairs
    3. Bounding boxes (blue rectangles) around detected persons
    
    Args:
        img: Input image to draw on (modified in-place)
        pose_result: Dictionary containing pose detection results:
            - keypoints: Array of shape (num_persons, 17, 2) with (x,y) coordinates
            - keypoint_scores: Array of shape (num_persons, 17) with confidence scores
            - bboxes: Array of shape (num_persons, 4) with bounding box coordinates
        connections: Sequence of (start_idx, end_idx) tuples defining skeleton structure
    """

    keypoints = pose_result["keypoints"]
    scores = pose_result["keypoint_scores"]
    bboxes = pose_result["bboxes"]

    for person_kps, person_bbox, scores in zip(keypoints, bboxes, scores):
        for kp, score in zip(person_kps, scores):
            if score > DRAW_CONFIDENCE_THRESHOLD:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        # Draw connections
        for start_idx, end_idx in connections:
            if (
                scores[start_idx] > DRAW_CONFIDENCE_THRESHOLD
                and scores[end_idx] > DRAW_CONFIDENCE_THRESHOLD
            ):
                start_point = tuple(np.int32(person_kps[start_idx]))
                end_point = tuple(np.int32(person_kps[end_idx]))
                cv2.line(img, start_point, end_point, (0, 255, 0), 2)

        x1, y1, x2, y2 = map(int, person_bbox)
        cv2.rectangle(
            img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2, lineType=cv2.LINE_AA
        )
