import cv2
import numpy as np
from typing import Optional, Sequence, Tuple


DRAW_CONFIDENCE_THRESHOLD = 0.3
COCO_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]
DEFAULT_ACTION_RECOGNITION_SPOT = (20, 30)


def draw_on_video(
    frames: Sequence[np.ndarray], pose_results: list, recognition_result: str
) -> Sequence[np.ndarray]:
    frames_ready = []
    for frame, pose_result in zip(frames, pose_results):
        draw_skeleton(frame, pose_result, COCO_CONNECTIONS)
        draw_recognition_result(frame, recognition_result)
        frames_ready.append(frame)

    return frames_ready


def draw_recognition_result(frame: np.ndarray, recognition_result: str) -> None:
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
    """Draw skeleton on image."""

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
