import cv2
import numpy as np
from typing import Sequence
import os.path as osp
import os

from posec3d_lib.utils.drawing import draw_on_video


def extract_frames(video_path: str, tmp_dir: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS)

    target_dir = osp.join(tmp_dir, osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, "img_{:06d}.jpg")

    frames = []
    frame_paths = []
    cnt = 0

    while True:
        rec, frame = cap.read()
        if not rec:
            break
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        cnt += 1

    return frame_paths, frames, fps


def save_video_to_file(
    frames: Sequence[np.ndarray],
    pose_result: dict,
    recognition_result: str,
    target_fps: int,
    ori_width: int,
    ori_height: int,
    output_video_path: str,
) -> bool:
    try:
        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_video_path, fourcc, target_fps, (ori_width, ori_height)
            )

        processed_frames = draw_on_video(frames, pose_result, recognition_result)
        for frame in processed_frames:
            writer.write(frame)
        return True

    except Exception as e:
        print(f"Something happened during video saving : {e}")
        return False
