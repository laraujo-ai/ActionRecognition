"""Video processing and recording utilities for action recognition visualization.

This module provides functions for extracting frames from videos, processing them
with pose and action overlays, and saving the results as output videos. Used for
creating visualization outputs of the action recognition pipeline.
"""

import cv2
import numpy as np
from typing import Sequence
import os.path as osp
import os

from posec3d_lib.utils.drawing import draw_on_video


def extract_frames(video_path: str, tmp_dir: str):
    """Extract frames from a video file and save them as individual images.
    
    Reads a video file and extracts all frames, saving each frame as a JPEG
    image in a temporary directory. Returns both the frame data in memory
    and the file paths for disk-based processing.
    
    Args:
        video_path: Path to the input video file to process
        tmp_dir: Temporary directory to store extracted frame images
        
    Returns:
        tuple: A 3-tuple containing:
            - frame_paths (list): List of paths to saved frame images
            - frames (list): List of frame arrays in memory
            - fps (float): Frame rate of the input video
            
    Note:
        Creates a subdirectory within tmp_dir named after the video file.
        Frame images are named with format 'img_{:06d}.jpg'.
    """
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
    """Save processed video frames with pose and action overlays to a video file.
    
    Takes a sequence of frames and overlays pose skeletons and action recognition
    results, then saves the result as an MP4 video file. This is used to create
    visualization outputs showing the complete action recognition pipeline results.
    
    Args:
        frames: Sequence of video frames as numpy arrays
        pose_result: Dictionary containing pose detection results for overlay
        recognition_result: Action class name to display as text overlay
        target_fps: Output video frame rate
        ori_width: Output video width in pixels
        ori_height: Output video height in pixels
        output_video_path: Path where the output video should be saved
        
    Returns:
        bool: True if video was saved successfully, False if an error occurred
        
    Note:
        Uses MP4V codec for output video encoding. Applies pose visualization
        and action text overlay to all frames before encoding.
    """
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
